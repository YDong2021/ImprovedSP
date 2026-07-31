import os
import argparse
import numpy as np
import random
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import clip
from sentence_transformers import SentenceTransformer
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import visformer
from data.dataloader import EpisodeSampler, MultiTrans
from data.dataset import DatasetWithTextLabel
from data.randaugment import RandAugmentMC
from utils import mean_confidence_interval


def main(args):
    # checkpoint and tensorboard dir
    args.tensorboard_dir = 'tensorboard/' + args.dataset + '/' + args.model + '/' + args.exp + '/'
    args.checkpoint_dir = 'checkpoint/' + args.dataset + '/' + args.model + '/' + args.exp + '/'
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.logger = SummaryWriter(args.tensorboard_dir)

    # prepare training and testing dataloader
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    train_aug = transforms.Compose([transforms.Resize(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    norm])
    if args.aug:
        train_aug = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        norm])
    if args.rand_aug:
        train_aug = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                        RandAugmentMC(2, 10, args.image_size),
                                        transforms.ToTensor(),
                                        norm])
    test_aug = transforms.Compose([transforms.Resize(int(args.image_size * 1.1)),
                                   transforms.CenterCrop(args.image_size),
                                   transforms.ToTensor(),
                                   norm])
    if args.aug_support > 1:
        aug = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                  # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  norm])
        test_aug = MultiTrans([test_aug] + [aug]*(args.aug_support-1))

    train_dataset = DatasetWithTextLabel(args.dataset, train_aug, split='train')
    n_episodes = args.train_episodes
    args.train_way = args.way if args.train_way == -1 else args.train_way
    if n_episodes == -1:
        n_episodes = int(len(train_dataset) / (args.train_way * (args.shot + 15)))
    episode_sampler = EpisodeSampler(train_dataset.dataset.targets,
                                     n_episodes,
                                     args.train_way,
                                     args.shot + 15, fix_seed=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=episode_sampler, num_workers=8)
    num_classes = len(train_dataset.dataset.classes)

    test_dataset = DatasetWithTextLabel(args.dataset, test_aug, split=args.split)
    episode_sampler = EpisodeSampler(test_dataset.dataset.targets, args.episodes, args.way, args.shot + 15)
    # num_workers=0: multiprocessing DataLoader workers are unstable in this container
    # (segfault / worker memory corruption). the test loop tolerates single-process loading.
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=episode_sampler, num_workers=0)

    # bi-level routing: the router is updated on episodes of held-out val classes. the base-class
    # training loss prefers the earliest injection layer (more layers left to reshape the features)
    # which measured as the worst layer on novel classes, so the routing decision needs a signal
    # that reflects transfer instead of fit
    val_loader, val_text = None, None
    if args.prompt_layer in ['dynamic', 'soft'] and args.router_split == 'val':
        val_dataset = DatasetWithTextLabel(args.dataset, train_aug, split='val')
        val_sampler = EpisodeSampler(val_dataset.dataset.targets, n_episodes,
                                     args.train_way, args.shot + 15, fix_seed=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=4)

    if args.nlp_model == 'clip':
        teacher, _ = clip.load("ViT-B/32", device='cuda:' + str(args.gpu))
        text_dim = 512
        # set the max text length
        if args.text_length != -1:
            teacher.context_length = args.text_length
            teacher.positional_embedding.data = teacher.positional_embedding.data[:args.text_length]
            for layer in teacher.transformer.resblocks:
                layer.attn_mask.data = layer.attn_mask.data[:args.text_length, :args.text_length]
    elif args.nlp_model == 'mpnet':
        teacher = SentenceTransformer('all-mpnet-base-v2', device=f'cuda:{args.gpu}')
        text_dim = 768
    elif args.nlp_model == 'glove':
        teacher = SentenceTransformer('average_word_embeddings_glove.6B.300d', device=f'cuda:{args.gpu}')
        text_dim = 300
    else:
        raise ValueError(f'unknown nlp_model: {args.nlp_model}')
    train_text = get_text_feature(teacher, train_dataset, args)
    test_text = get_text_feature(teacher, test_dataset, args)
    if val_loader is not None:
        val_text = get_text_feature(teacher, val_dataset, args)
    if args.eqnorm:
        if args.nlp_model in ['mpnet', 'glove']:
            # the bert features have been normalized to unit length. use the avg norm of clip text features
            avg_length = 9.
        else:
            avg_length = (train_text ** 2).sum(-1).sqrt().mean().item()
        train_text = F.normalize(train_text, dim=-1) * avg_length
        test_text = F.normalize(test_text, dim=-1) * avg_length
        if val_text is not None:
            val_text = F.normalize(val_text, dim=-1) * avg_length

    if args.model == 'visformer-t':
        student = visformer.visformer_tiny(num_classes=num_classes)
    elif args.model == 'visformer-t-84':
        student = visformer.visformer_tiny_84(num_classes=num_classes)
    else:
        raise ValueError(f'unknown model: {args.model}')

    feature_dim = 384
    if 2 <= args.stage < 3:
        feature_dim = 192
    if args.projector == 'linear':
        student.t2i = torch.nn.Linear(text_dim, feature_dim, bias=False)
    elif args.projector == 'mlp':
        student.t2i = torch.nn.Sequential(torch.nn.Linear(text_dim, text_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(text_dim, feature_dim, bias=False))
    elif args.projector == 'mlp3':
        student.t2i = torch.nn.Sequential(torch.nn.Linear(text_dim, text_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(text_dim, text_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(text_dim, feature_dim, bias=False))

    if 'channel' in args.prompt_mode:
        student.t2i2 = torch.nn.Linear(text_dim, feature_dim, bias=False)
        student.se_block = torch.nn.Sequential(torch.nn.Linear(feature_dim*2, feature_dim, bias=True),
                                               torch.nn.Sigmoid(),
                                               torch.nn.Linear(feature_dim, feature_dim),
                                               torch.nn.Sigmoid(),)

    if args.prompt_layer in ['dynamic', 'soft']:
        # router over the stage3 layers, class-level decision from text features.
        # the input LayerNorm strips the dominant mean direction of the text features and keeps
        # the logits at a sane scale (eqnorm scales the raw features to norm ~9)
        num_inject_layers = len(student.stage3)
        student.router = torch.nn.Sequential(torch.nn.LayerNorm(text_dim),
                                             torch.nn.Linear(text_dim, 128),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(128, num_inject_layers))
        # warm start: bias the router towards layer 3.2 (index 2, the known-best fixed layer).
        # with the straight-through gate the selection probability is softmax(logits), so a bias
        # of 2.0 routes ~71% of the classes to layer 3.2 at the beginning
        with torch.no_grad():
            student.router[3].bias.zero_()
            student.router[3].bias[2] = args.router_warm_bias

    student = student.cuda(args.gpu)

    optim_params_id = [id(param) for param in student.t2i.parameters()]
    if 'channel' in args.prompt_mode:
        optim_params_id += [id(param) for param in student.t2i2.parameters()]  # se_block is not included. use smaller lr for se_block
        # optim_params_id += [id(param) for param in student.se_block.parameters()]
    router_params_id = []
    if args.prompt_layer in ['dynamic', 'soft']:
        # the router gets its own smaller lr: it is a gating net and runaway logits collapse the routing
        router_params_id = [id(param) for param in student.router.parameters()]
    optim_params = [param for param in student.parameters() if id(param) in optim_params_id]
    router_params = [param for param in student.parameters() if id(param) in router_params_id]
    other_params = [param for param in student.parameters()
                    if id(param) not in optim_params_id + router_params_id]
    if args.optim == 'sgd':
        optim = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == 'adamw':
        param_groups = [{'params': optim_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
                        {'params': other_params, 'lr': args.encoder_lr}]
        optim = torch.optim.AdamW(param_groups, weight_decay=5e-2)
    else:
        raise ValueError(f'unknown optim: {args.optim}')

    # the router has its own optimizer so that it can be updated on a different episode stream
    # than the rest of the network. no weight decay: the routing signal is weak (all stage3 layers
    # work almost equally well), so decay would shrink the logits to a constant uniform gate
    router_optim = None
    if router_params:
        router_optim = torch.optim.AdamW(router_params, lr=args.router_lr, weight_decay=args.router_wd)

    if args.resume:
        args.init = args.resume
    if args.init:
        checkpoint = torch.load(args.init, map_location=f'cuda:{args.gpu}')
        student.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        raise ValueError('must provide pre-trained model')

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu}')
        student.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        if router_optim is not None and 'router_optimizer' in checkpoint:
            router_optim.load_state_dict(checkpoint['router_optimizer'])
        start_epoch = checkpoint['epoch']
        print(f'load checkpoint at epoch {start_epoch}')

    if args.test:
        test(test_text, student, test_loader, 0, args)
        return

    best_acc = 0.
    for epoch in range(start_epoch, args.epochs):
        train(train_text, student, train_loader, optim, epoch, args,
              router_optim=router_optim, val_loader=val_loader, val_text=val_text)

        if (epoch + 1) % args.test_freq == 0:
            acc = test(test_text, student, test_loader, epoch, args)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': student.state_dict(),
            'optimizer': optim.state_dict(),
        }
        if router_optim is not None:
            checkpoint['router_optimizer'] = router_optim.state_dict()
        torch.save(checkpoint, args.checkpoint_dir + f'checkpoint_epoch_latest.pth')
        if (epoch + 1) % args.save_freq == 0:
            torch.save(checkpoint, args.checkpoint_dir + f'checkpoint_epoch_{epoch + 1:03d}.pth')
        if (epoch + 1) % args.test_freq == 0 and acc > best_acc:
            best_acc = acc
            torch.save(checkpoint, args.checkpoint_dir + f'checkpoint_epoch_best.pth')


def get_text_feature(teacher, dataset, args):
    class_idx = dataset.dataset.classes
    idx2text = dataset.idx2text
    if args.no_template:
        text = [idx2text[idx] for idx in class_idx]
    else:
        text = ['A photo of ' + idx2text[idx] for idx in class_idx]

    teacher.eval()
    if args.nlp_model == 'clip':
        text_token = clip.tokenize(text).cuda(args.gpu)
        if args.text_length != -1:
            text_token = text_token[:, :args.text_length]
        with torch.no_grad():
            text_feature = teacher.encode_text(text_token)
            text_feature = text_feature.float()
    else:
        with torch.no_grad():
            text_feature = teacher.encode(text)
            text_feature = torch.tensor(text_feature).cuda(args.gpu)

    return text_feature


def route_gate(student, text_features, args, tau=None, training=False):
    logits = student.router(text_features)
    if args.prompt_layer == 'soft':
        # plan B: every stage3 layer is injected with a softmax weight. train and test are identical
        return logits, logits.softmax(dim=-1)
    if training:
        # straight-through: the forward pass is one-hot, matching the test-time hard routing,
        # while the gradient flows through the soft Gumbel-Softmax. with hard=True the selection
        # probability is exactly softmax(logits), independent of tau
        return logits, F.gumbel_softmax(logits, tau=tau, hard=not args.soft_gate, dim=-1)
    # test-time routing: one-hot on the argmax layer, no Gumbel sampling
    return logits, F.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[-1]).float()


def router_val_step(student, val_text, episode, args, tau, progress):
    """one router update signal computed on an episode of held-out val classes. the returned loss
    reaches the router parameters through the support features only; the caller is responsible for
    clearing the gradients that this backward leaves on the rest of the network"""
    image = episode[0].cuda(args.gpu)
    glabels = episode[1].cuda(args.gpu)
    labels = torch.arange(args.train_way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)

    image = image.view(args.train_way, args.shot + 15, *image.shape[1:])
    sup, que = image[:, :args.shot].contiguous(), image[:, args.shot:].contiguous()
    sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

    glabels = glabels.view(args.train_way, args.shot + 15)[:, :args.shot]
    glabels = glabels.contiguous().view(-1)
    text_features = val_text[glabels]

    logits, gate = route_gate(student, text_features, args, tau=tau, training=True)
    _, sup_im_features = student.forward_with_dynamic_prompt(sup, text_features, gate, args)
    sup_im_features = sup_im_features.view(args.train_way, args.shot, -1).mean(dim=1)
    _, que_im_features = student(que)

    sim = F.normalize(que_im_features, dim=-1) @ F.normalize(sup_im_features, dim=-1).t()
    loss = F.cross_entropy(sim / args.t, labels)
    if args.balance_reg > 0:
        p = logits.softmax(dim=-1)
        p_mean = p.mean(0)
        marginal_entropy = -(p_mean * p_mean.clamp_min(1e-9).log()).sum()
        loss = loss - args.balance_reg * (1 - progress) * marginal_entropy
    return loss, logits.detach()


def train(text, student, train_loader, optim, epoch, args,
          router_optim=None, val_loader=None, val_text=None):
    student.train()
    losses = 0.
    accs = 0.
    route_probs = 0.
    route_sel = 0.
    route_arg = 0.
    route_conf = 0.
    tau = 0.
    # bi-level mode: the router is driven by val-class episodes instead of the train loss
    router_on_val = router_optim is not None and val_loader is not None
    # single-level mode: the router is a normal part of the graph and is updated by the train loss.
    # its parameters live only in router_optim, so this optimizer has to be stepped here as well,
    # otherwise the router stays frozen at its warm-start initialisation
    router_on_train = router_optim is not None and not router_on_val
    val_iter = iter(val_loader) if router_on_val else None
    val_losses, val_steps = 0., 0
    val_route_arg = 0.
    val_route_conf = 0.
    for idx, episode in enumerate(train_loader):
        image = episode[0].cuda(args.gpu)  # way * (shot+15)
        glabels = episode[1].cuda(args.gpu)
        labels = torch.arange(args.train_way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)

        image = image.view(args.train_way, args.shot+15, *image.shape[1:])
        sup, que = image[:, :args.shot].contiguous(), image[:, args.shot:].contiguous()
        sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

        glabels = glabels.view(args.train_way, args.shot+15)[:, :args.shot]
        glabels = glabels.contiguous().view(-1)
        text_features = text[glabels]
        if args.prompt_layer in ['dynamic', 'soft']:
            # linear tau annealing over the whole training
            progress = epoch / max(args.epochs - 1, 1)
            tau = args.gumbel_tau_start + (args.gumbel_tau_end - args.gumbel_tau_start) * progress
            logits, gate = route_gate(student, text_features, args, tau=tau, training=True)
            # route_probs: selection probability of the router. route_sel: the gate weights that
            # were actually applied. route_arg: the layer the test-time hard routing would pick.
            # route_conf: per-class confidence. a value near 1/num_layers means the router is
            # not discriminating between the layers at all, whatever the averaged probs look like
            p_det = logits.softmax(dim=-1).detach()
            route_probs = route_probs + p_det.mean(0)
            route_sel = route_sel + gate.mean(0).detach()
            route_arg = route_arg + F.one_hot(p_det.argmax(dim=-1), p_det.shape[-1]).float().mean(0)
            route_conf = route_conf + p_det.max(dim=-1).values.mean()
            if router_on_val:
                # the network is trained through a fixed routing decision. letting the train loss
                # touch the router is exactly what biased it towards the earliest (worst
                # generalizing) injection layer
                gate = gate.detach()
                if args.train_route == 'uniform':
                    # one-shot NAS style path sampling: the weights are trained through a uniformly
                    # drawn injection layer so that every candidate path stays usable. driving the
                    # training with the router instead makes it collapse onto one layer, and the
                    # paths that the unseen classes get routed to at test time go stale
                    pick = torch.randint(gate.shape[-1], (gate.shape[0],), device=gate.device)
                    gate = F.one_hot(pick, gate.shape[-1]).float()
            _, sup_im_features = student.forward_with_dynamic_prompt(sup, text_features, gate, args)
        elif args.prompt_mode == 'spatial':
            text_features = student.t2i(text_features)
            _, sup_im_features = student.forward_with_semantic_prompt(sup, text_features, args)
        else:
            _, sup_im_features = student.forward_with_semantic_prompt_channel(sup, text_features, args)

        sup_im_features = sup_im_features.view(args.train_way, args.shot, -1).mean(dim=1)

        _, que_im_features = student(que)

        sim = F.normalize(que_im_features, dim=-1) @ F.normalize(sup_im_features, dim=-1).t()
        loss = F.cross_entropy(sim / args.t, labels)
        if args.prompt_layer in ['dynamic', 'soft'] and not router_on_val:
            p = logits.softmax(dim=-1)
            if args.balance_reg > 0:
                # load balancing: keep the batch-marginal routing diverse so that the router cannot
                # send every class to the same layer, while each single class stays free to be
                # confident. decayed to 0 so the routing can settle late in training
                p_mean = p.mean(0)
                marginal_entropy = -(p_mean * p_mean.clamp_min(1e-9).log()).sum()
                loss = loss - args.balance_reg * (1 - progress) * marginal_entropy
            if args.entropy_reg > 0:
                # per-class entropy bonus. off by default: it fights the confident per-class
                # decision that the router is supposed to learn
                entropy = -(p * p.clamp_min(1e-9).log()).sum(dim=-1).mean()
                loss = loss - args.entropy_reg * (1 - progress) * entropy
        losses += loss.item()
        _, pred = sim.max(-1)
        accs += labels.eq(pred).sum().float().item() / labels.shape[0]

        optim.zero_grad()
        if router_on_train:
            router_optim.zero_grad()
        loss.backward()
        optim.step()
        if router_on_train:
            router_optim.step()

        if router_on_val and idx % args.router_every == 0:
            try:
                val_episode = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_episode = next(val_iter)
            r_loss, r_logits = router_val_step(student, val_text, val_episode, args, tau, progress)
            optim.zero_grad()
            router_optim.zero_grad()
            r_loss.backward()
            router_optim.step()
            # this backward also filled the gradients of the backbone and the projectors. discard
            # them so that the val episode never leaks into the parameters owned by optim
            optim.zero_grad()
            router_optim.zero_grad()
            p_val = r_logits.softmax(dim=-1)
            val_route_arg = val_route_arg + F.one_hot(p_val.argmax(dim=-1), p_val.shape[-1]).float().mean(0)
            val_route_conf = val_route_conf + p_val.max(dim=-1).values.mean()
            val_losses += r_loss.item()
            val_steps += 1

        if idx % args.print_step == 0 or idx == len(train_loader) - 1:
            print_string = f'Train epoch: {epoch}, step: {idx:3d}, loss: {losses / (idx + 1):.4f}, acc: {accs * 100 / (idx + 1):.2f}'
            print(print_string)
    args.logger.add_scalar('train/loss', losses / len(train_loader), epoch)
    args.logger.add_scalar('train/acc', accs / len(train_loader), epoch)
    if args.prompt_layer in ['dynamic', 'soft']:
        route_probs = route_probs / len(train_loader)
        route_sel = route_sel / len(train_loader)
        route_arg = route_arg / len(train_loader)
        route_conf = route_conf / len(train_loader)
        print('router probs:', [f'{p:.3f}' for p in route_probs.tolist()],
              '| argmax:', [f'{p:.3f}' for p in route_arg.tolist()],
              f'| conf: {route_conf:.3f} | tau: {tau:.3f}')
        for l in range(route_probs.shape[-1]):
            args.logger.add_scalar(f'train/route_p{l}', route_probs[l].item(), epoch)
            args.logger.add_scalar(f'train/route_gate{l}', route_sel[l].item(), epoch)
            args.logger.add_scalar(f'train/route_argmax{l}', route_arg[l].item(), epoch)
        args.logger.add_scalar('train/route_conf', route_conf.item(), epoch)
        args.logger.add_scalar('train/gumbel_tau', tau, epoch)
    if val_steps > 0:
        # the val classes are unseen, so this argmax distribution is the honest preview of how the
        # router will behave on the novel classes at test time
        val_route_arg = val_route_arg / val_steps
        val_route_conf = val_route_conf / val_steps
        print(f'val router loss: {val_losses / val_steps:.4f}',
              '| argmax:', [f'{p:.3f}' for p in val_route_arg.tolist()],
              f'| conf: {val_route_conf:.3f}')
        for l in range(val_route_arg.shape[-1]):
            args.logger.add_scalar(f'router_val/argmax{l}', val_route_arg[l].item(), epoch)
        args.logger.add_scalar('router_val/conf', val_route_conf.item(), epoch)
        args.logger.add_scalar('router_val/loss', val_losses / val_steps, epoch)


def test(text, student, test_loader, epoch, args):
    student.eval()
    accs = []
    with torch.no_grad():
        for episode in test_loader:
            if args.aug_support == 1:
                # use prototype classifier
                image = episode[0].cuda(args.gpu)  # way * (shot+15)
                glabels = episode[1].cuda(args.gpu)
                labels = torch.arange(args.way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)

                image = image.view(args.way, args.shot + 15, *image.shape[1:])
                sup, que = image[:, :args.shot].contiguous(), image[:, args.shot:].contiguous()
                sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

                glabels = glabels.view(args.way, args.shot + 15)[:, :args.shot]
                glabels = glabels.contiguous().view(-1)
                text_features = text[glabels]
                if args.prompt_layer in ['dynamic', 'soft']:
                    _, gate = route_gate(student, text_features, args)
                    _, sup_im_features = student.forward_with_dynamic_prompt(sup, text_features, gate, args)
                elif args.prompt_mode == 'spatial':
                    text_features = student.t2i(text_features)
                    _, sup_im_features = student.forward_with_semantic_prompt(sup, text_features, args)
                else:
                    _, sup_im_features = student.forward_with_semantic_prompt_channel(sup, text_features, args)
                _, que_im_features = student(que)

                if args.test_classifier == 'prototype':
                    sup_im_features = sup_im_features.view(args.way, args.shot, -1).mean(dim=1)
                    sim = F.normalize(que_im_features, dim=-1) @ F.normalize(sup_im_features, dim=-1).t()
                    _, pred = sim.max(-1)
                elif args.test_classifier == 'fc':
                    x_train = F.normalize(sup_im_features, dim=-1).cpu().numpy()
                    y_train = torch.arange(args.way).unsqueeze(-1).repeat(1, args.shot).view(-1).numpy()
                    # x_test = F.normalize(que_im_features, dim=-1).cpu().numpy()
                    x_test = que_im_features.cpu().numpy()
                    from sklearn.linear_model import LogisticRegression
                    clf = LogisticRegression(penalty='l2',
                                             random_state=0,
                                             C=1,
                                             solver='lbfgs',
                                             max_iter=1000,
                                             multi_class='multinomial')
                    clf.fit(x_train, y_train)
                    pred = clf.predict(x_test)
                    pred = torch.tensor(pred).cuda(args.gpu)

            elif args.aug_support > 1:
                # use logistic regression classifier
                image = torch.cat(episode[0]).cuda(args.gpu)  # aug_support * way * (shot+15)
                glabels = episode[1].cuda(args.gpu)
                labels = torch.arange(args.way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)

                image = image.view(args.aug_support, args.way, args.shot + 15, *image.shape[1:])
                sup = image[:, :, :args.shot].contiguous().view(-1, *image.shape[3:])
                que = image[0, :, args.shot:].contiguous().view(-1, *image.shape[3:])

                glabels = glabels.view(args.way, args.shot + 15)[:, :args.shot]
                glabels = glabels.unsqueeze(0).repeat(args.aug_support, 1, 1).contiguous().view(-1)
                text_features = text[glabels]
                # text_features = student.t2i(text_features)
                # _, sup_im_features = student.forward_with_semantic_prompt(sup, text_features, args)
                if args.prompt_layer in ['dynamic', 'soft']:
                    _, gate = route_gate(student, text_features, args)
                    _, sup_im_features = student.forward_with_dynamic_prompt(sup, text_features, gate, args)
                elif args.prompt_mode == 'spatial':
                    text_features = student.t2i(text_features)
                    _, sup_im_features = student.forward_with_semantic_prompt(sup, text_features, args)
                else:
                    _, sup_im_features = student.forward_with_semantic_prompt_channel(sup, text_features, args)

                _, que_im_features = student(que)

                if args.test_classifier == 'prototype':
                    sup_im_features = sup_im_features.view(args.aug_support, args.way, args.shot, -1).mean(dim=0).mean(dim=1)
                    sim = F.normalize(que_im_features, dim=-1) @ F.normalize(sup_im_features, dim=-1).t()
                    _, pred = sim.max(-1)
                elif args.test_classifier == 'fc':
                    x_train = F.normalize(sup_im_features, dim=-1).cpu().numpy()
                    y_train = torch.arange(args.way).unsqueeze(0).unsqueeze(-1).repeat(args.aug_support, 1, args.shot).view(-1).numpy()
                    x_test = F.normalize(que_im_features, dim=-1).cpu().numpy()
                    from sklearn.linear_model import LogisticRegression
                    clf = LogisticRegression(penalty='l2',
                                             random_state=0,
                                             C=1.0,
                                             solver='lbfgs',
                                             max_iter=1000,
                                             multi_class='multinomial')
                    clf.fit(x_train, y_train)
                    pred = clf.predict(x_test)
                    pred = torch.tensor(pred).cuda(args.gpu)

            acc = labels.eq(pred).sum().float().item() / labels.shape[0]
            accs.append(acc)

    m, h = mean_confidence_interval(accs)
    print(f'Test epoch: {epoch}, test acc: {m * 100:.2f}+-{h * 100:.2f}')
    args.logger.add_scalar('test/acc', m * 100, epoch)

    return m


if __name__ == '__main__':
    # use 'spawn' instead of the default 'fork' to avoid segfault when
    # multiprocessing DataLoader workers start in this container environment
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='debug')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100'])
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--image_size', type=int, default=224, choices=[224, 84])
    parser.add_argument('--aug', action='store_true', default=True)
    parser.add_argument('--rand_aug', action='store_true')
    parser.add_argument('--aug_support', type=int, default=1)
    parser.add_argument('--model', type=str, default='visformer-t', choices=['visformer-t', 'visformer-t-84'])
    parser.add_argument('--nlp_model', type=str, default='clip', choices=['clip', 'glove', 'mpnet'])
    parser.add_argument('--prompt_mode', type=str, default='spatial+channel', choices=['spatial', 'channel', 'spatial+channel'])
    parser.add_argument('--prompt_layer', type=str, default='fixed', choices=['fixed', 'dynamic', 'soft'])
    parser.add_argument('--gumbel_tau_start', type=float, default=5.)
    parser.add_argument('--gumbel_tau_end', type=float, default=0.5)
    parser.add_argument('--soft_gate', action='store_true')
    parser.add_argument('--router_lr', type=float, default=5e-4)
    parser.add_argument('--router_wd', type=float, default=0.)
    parser.add_argument('--router_warm_bias', type=float, default=2.)
    parser.add_argument('--balance_reg', type=float, default=0.01)
    parser.add_argument('--entropy_reg', type=float, default=0.)
    parser.add_argument('--router_split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--router_every', type=int, default=4)
    parser.add_argument('--train_route', type=str, default='uniform', choices=['router', 'uniform'])
    parser.add_argument('--no_template', action='store_true')
    parser.add_argument('--eqnorm', action='store_true', default=True)
    parser.add_argument('--stage', type=float, default=3.2, choices=[2, 2.1, 2.2, 2.3, 3, 3.1, 3.2, 3.3])
    parser.add_argument('--projector', type=str, default='linear', choices=['linear', 'mlp', 'mlp3'])
    parser.add_argument('--avg', type=str, default='all', choices=['all', 'patch', 'head'])
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--optim', type=str, default='adamw', choices=['sgd', 'adamw'])
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--encoder_lr', type=float, default=1e-6)
    parser.add_argument('--init', type=str, default='checkpoint/miniImageNet/visformer-t/pre-train/checkpoint_epoch_800.pth')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--text_length', type=int, default=20)
    parser.add_argument('--train_way', type=int, default=-1)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_episodes', type=int, default=-1)
    parser.add_argument('--episodes', type=int, default=600)
    parser.add_argument('--test_classifier', type=str, default='prototype', choices=['prototype', 'fc'])
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=20)

    args = parser.parse_args()
    if args.seed >= 0:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    main(args)

