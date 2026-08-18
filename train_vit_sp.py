import os
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
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


class LayerSelectNet(nn.Module):
    """per-layer injection decision for stage3. shared trunk encodes the GAP visual features
    and the text prompt, each stage3 layer owns a small head producing one injection logit.
    decisions are consumed sequentially inside forward_with_selective_prompt: the first sigmoid
    above 0.5 injects and stops the process, so exactly one layer is selected per sample"""
    def __init__(self, feature_dim, text_dim, num_layers, hidden=64, warm_bias=2.):
        super().__init__()
        self.vis_fc = nn.Linear(feature_dim, hidden)
        self.text_fc = nn.Linear(text_dim, hidden)
        self.act = nn.ReLU(inplace=True)
        self.heads = nn.ModuleList([nn.Linear(hidden * 2, 1) for _ in range(num_layers)])
        # warm start: zero the head weights so the initial logit equals the bias for every
        # input (random head weights would amplify the trunk activations and drown the bias,
        # making the warm start nominal). a tiny noise breaks the exact zero-weight symmetry.
        # p0, p1 < 0.5 and p2 > 0.5, so the initial hard decision injects layer 2
        # (the known-best fixed layer 3.2) for every sample
        with torch.no_grad():
            for l, head in enumerate(self.heads):
                head.weight.mul_(1e-3)
                head.bias.fill_(-warm_bias)
            self.heads[2].bias.fill_(warm_bias)

    def forward(self, v, t, layer):
        h = torch.cat([self.vis_fc(v), self.text_fc(t)], dim=-1)
        return self.heads[layer](self.act(h)).view(-1)


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

    if args.use_caption:
        # instance-level: per-image CLIP features pre-computed from LLaVA captions
        # (generate_captions.py -> captions.json -> CLIP encoding -> *_clip_features.pt).
        # features are indexed by the global sample index returned by the dataset, encoded
        # with the full 77-token CLIP context, so no teacher model / eqnorm is needed here
        text_dim = 512
        train_text = torch.load(f'data/captions/{args.dataset}_train_clip_features.pt').cuda(args.gpu)
        test_text = torch.load(f'data/captions/{args.dataset}_{args.split}_clip_features.pt').cuda(args.gpu)
        print(f'Loaded caption features: train {train_text.shape}, test {test_text.shape}')
    else:
        # class-level: original text template approach
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
        if args.eqnorm:
            if args.nlp_model in ['mpnet', 'glove']:
                # the bert features have been normalized to unit length. use the avg norm of clip text features
                avg_length = 9.
            else:
                avg_length = (train_text ** 2).sum(-1).sqrt().mean().item()
            train_text = F.normalize(train_text, dim=-1) * avg_length
            test_text = F.normalize(test_text, dim=-1) * avg_length

    if args.model == 'visformer-t':
        student = visformer.visformer_tiny(num_classes=num_classes)
    elif args.model == 'visformer-t-84':
        student = visformer.visformer_tiny_84(num_classes=num_classes)
    else:
        raise ValueError(f'unknown model: {args.model}')

    feature_dim = 384
    if 2 <= args.stage < 3:
        feature_dim = 192
    if args.prompt_layer == 'selective':
        # selective injection always happens inside stage3 regardless of args.stage
        feature_dim = 384
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
    elif args.projector == 'bottleneck':
        # bottleneck adapter: down-project to a low-dim latent space and back up
        student.t2i = torch.nn.Sequential(torch.nn.Linear(text_dim, args.bottleneck_dim),
                                          torch.nn.GELU(),
                                          torch.nn.Linear(args.bottleneck_dim, feature_dim, bias=False))

    if 'channel' in args.prompt_mode:
        student.t2i2 = torch.nn.Linear(text_dim, feature_dim, bias=False)
        student.se_block = torch.nn.Sequential(torch.nn.Linear(feature_dim*2, feature_dim, bias=True),
                                               torch.nn.Sigmoid(),
                                               torch.nn.Linear(feature_dim, feature_dim),
                                               torch.nn.Sigmoid(),)

    decision_net = None
    if args.prompt_layer == 'selective':
        decision_net = LayerSelectNet(feature_dim, text_dim, len(student.stage3),
                                      warm_bias=args.decision_warm_bias)

    student = student.cuda(args.gpu)
    if decision_net is not None:
        decision_net = decision_net.cuda(args.gpu)

    optim_params_id = [id(param) for param in student.t2i.parameters()]
    if 'channel' in args.prompt_mode:
        optim_params_id += [id(param) for param in student.t2i2.parameters()]  # se_block is not included. use smaller lr for se_block
        # optim_params_id += [id(param) for param in student.se_block.parameters()]
    optim_params = [param for param in student.parameters() if id(param) in optim_params_id]
    other_params = [param for param in student.parameters()
                    if id(param) not in optim_params_id]
    if args.optim == 'sgd':
        all_params = list(student.parameters()) + (list(decision_net.parameters()) if decision_net is not None else [])
        optim = torch.optim.SGD(all_params, lr=args.lr, momentum=0.9)
    elif args.optim == 'adamw':
        param_groups = [{'params': optim_params, 'lr': args.lr, 'weight_decay': args.weight_decay}]
        if decision_net is not None:
            # the decision net gets its own group with a reduced lr: a fast head would race
            # ahead of the backbone and collapse the selection onto one layer before the
            # backbone has adapted to multi-layer injection
            param_groups.append({'params': list(decision_net.parameters()),
                                 'lr': args.lr * args.decision_lr_mult,
                                 'weight_decay': args.weight_decay})
        param_groups.append({'params': other_params, 'lr': args.encoder_lr})
        optim = torch.optim.AdamW(param_groups, weight_decay=5e-2)
    else:
        raise ValueError(f'unknown optim: {args.optim}')

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
        if decision_net is not None and 'decision_net' in checkpoint:
            decision_net.load_state_dict(checkpoint['decision_net'])
        start_epoch = checkpoint['epoch']
        print(f'load checkpoint at epoch {start_epoch}')

    if args.test:
        test(test_text, student, decision_net, test_loader, 0, args)
        return

    best_acc = 0.
    for epoch in range(start_epoch, args.epochs):
        train(train_text, student, decision_net, train_loader, optim, epoch, args)

        if (epoch + 1) % args.test_freq == 0:
            acc = test(test_text, student, decision_net, test_loader, epoch, args)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': student.state_dict(),
            'optimizer': optim.state_dict(),
        }
        if decision_net is not None:
            checkpoint['decision_net'] = decision_net.state_dict()
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


def train(text, student, decision_net, train_loader, optim, epoch, args):
    student.train()
    if decision_net is not None:
        decision_net.train()
        # decision warmup: freeze the heads for the first epochs so the backbone and the
        # projectors first adapt to the warm-start injection (layer 2, i.e. fixed 3.2);
        # otherwise the heads race ahead and collapse the selection onto one layer.
        # requires_grad (not lr=0) is needed because AdamW still moves zero-grad params
        # through weight decay
        frozen = epoch < args.decision_warmup_epochs
        for p in decision_net.parameters():
            p.requires_grad_(not frozen)
        if epoch == 0 or epoch == args.decision_warmup_epochs:
            print(f'decision heads {"frozen (warmup)" if frozen else "unfrozen"} at epoch {epoch}')
    losses = 0.
    align_losses = 0.
    preserve_losses = 0.
    accs = 0.
    num_layers = len(student.stage3)
    select_hist = torch.zeros(num_layers)
    align_labels = torch.arange(args.train_way).cuda(args.gpu)
    for idx, episode in enumerate(train_loader):
        image = episode[0].cuda(args.gpu)  # way * (shot+15)
        glabels = episode[1].cuda(args.gpu)
        labels = torch.arange(args.train_way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)

        image = image.view(args.train_way, args.shot+15, *image.shape[1:])
        sup, que = image[:, :args.shot].contiguous(), image[:, args.shot:].contiguous()
        sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

        if args.use_caption:
            # per-image: use global sample indices to look up caption features
            indices = episode[2].cuda(args.gpu)
            indices = indices.view(args.train_way, args.shot + 15)[:, :args.shot]
            indices = indices.contiguous().view(-1)
            text_features = text[indices]
        else:
            # per-class: use class labels
            glabels = glabels.view(args.train_way, args.shot+15)[:, :args.shot]
            glabels = glabels.contiguous().view(-1)
            text_features = text[glabels]

        # project the text features once: prompt_feats is reused both as the spatial prompt
        # (passed to the forward as prompt1 to avoid a second t2i call) and as the source of
        # the InfoNCE alignment loss
        prompt_feats = student.t2i(text_features)

        if args.prompt_layer == 'selective':
            _, sup_im_features, selection, p_hist = student.forward_with_selective_prompt(
                sup, text_features, decision_net, args, prompt1=prompt_feats)
            # selection histogram: how often each stage3 layer ends up being the injection layer.
            # a high share of the last layer means the fallback keeps firing, i.e. the first
            # three decision heads are all too conservative
            select_hist = select_hist + torch.bincount(selection, minlength=num_layers).float()
        elif args.prompt_mode == 'spatial':
            _, sup_im_features = student.forward_with_semantic_prompt(sup, prompt_feats, args)
        else:
            _, sup_im_features = student.forward_with_semantic_prompt_channel(sup, text_features, args,
                                                                              prompt1=prompt_feats)

        sup_protos = sup_im_features.view(args.train_way, args.shot, -1).mean(dim=1)

        _, que_im_features = student(que)

        sim = F.normalize(que_im_features, dim=-1) @ F.normalize(sup_protos, dim=-1).t()
        loss = F.cross_entropy(sim / args.t, labels)

        if args.prompt_layer == 'selective':
            # batch-level load balancing: the STE classification signal alone rewards early
            # injection and collapses the selection onto one layer; maximizing the entropy of
            # the mean per-layer activation keeps all layers in use
            q = p_hist.mean(dim=1)
            entropy = -(q * (q + 1e-6).log()).sum()
            loss = loss - args.select_entropy_w * entropy

        # modality alignment: pull class prompt embeddings toward their visual prototypes (InfoNCE)
        if args.align_weight > 0:
            prompt_protos = prompt_feats.view(args.train_way, args.shot, -1).mean(dim=1)
            align_sim = F.normalize(prompt_protos, dim=-1) @ F.normalize(sup_protos, dim=-1).t()
            align_loss = F.cross_entropy(align_sim / args.t, align_labels)
            loss = loss + args.align_weight * align_loss
            align_losses += align_loss.item()

        # feature preservation: keep prompt-injected support features close to clean backbone features
        if args.preserve_weight > 0:
            with torch.no_grad():
                _, sup_clean_features = student(sup)
            preserve_loss = (1 - F.cosine_similarity(sup_im_features, sup_clean_features, dim=-1)).mean()
            loss = loss + args.preserve_weight * preserve_loss
            preserve_losses += preserve_loss.item()

        losses += loss.item()
        _, pred = sim.max(-1)
        accs += labels.eq(pred).sum().float().item() / labels.shape[0]

        optim.zero_grad()
        loss.backward()
        optim.step()

        if idx % args.print_step == 0 or idx == len(train_loader) - 1:
            print_string = f'Train epoch: {epoch}, step: {idx:3d}, loss: {losses / (idx + 1):.4f}, acc: {accs * 100 / (idx + 1):.2f}'
            print(print_string)
    args.logger.add_scalar('train/loss', losses / len(train_loader), epoch)
    args.logger.add_scalar('train/acc', accs / len(train_loader), epoch)
    if args.align_weight > 0:
        args.logger.add_scalar('train/align_loss', align_losses / len(train_loader), epoch)
    if args.preserve_weight > 0:
        args.logger.add_scalar('train/preserve_loss', preserve_losses / len(train_loader), epoch)
    if args.prompt_layer == 'selective':
        select_freq = select_hist / select_hist.sum()
        print('selection freq:', [f'{p:.3f}' for p in select_freq.tolist()],
              f'| fallback: {select_freq[-1]:.3f}')
        for l in range(num_layers):
            args.logger.add_scalar(f'train/select_l{l}', select_freq[l].item(), epoch)


def test(text, student, decision_net, test_loader, epoch, args):
    student.eval()
    if decision_net is not None:
        decision_net.eval()
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

                if args.use_caption:
                    indices = episode[2].cuda(args.gpu)
                    indices = indices.view(args.way, args.shot + 15)[:, :args.shot]
                    indices = indices.contiguous().view(-1)
                    text_features = text[indices]
                else:
                    glabels = glabels.view(args.way, args.shot + 15)[:, :args.shot]
                    glabels = glabels.contiguous().view(-1)
                    text_features = text[glabels]
                if args.prompt_layer == 'selective':
                    _, sup_im_features, _, _ = student.forward_with_selective_prompt(sup, text_features, decision_net, args)
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

                if args.use_caption:
                    indices = episode[2].cuda(args.gpu)
                    indices = indices.view(args.way, args.shot + 15)[:, :args.shot]
                    indices = indices.unsqueeze(0).repeat(args.aug_support, 1, 1).contiguous().view(-1)
                    text_features = text[indices]
                else:
                    glabels = glabels.view(args.way, args.shot + 15)[:, :args.shot]
                    glabels = glabels.unsqueeze(0).repeat(args.aug_support, 1, 1).contiguous().view(-1)
                    text_features = text[glabels]
                if args.prompt_layer == 'selective':
                    _, sup_im_features, _, _ = student.forward_with_selective_prompt(sup, text_features, decision_net, args)
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
    # idea 2: per-image LLaVA caption features instead of class-level text templates.
    # disable with --no-use_caption for ablation (falls back to the nlp_model template path)
    parser.add_argument('--use_caption', action=argparse.BooleanOptionalAction, default=True,
                        help='Use per-image LLaVA captions instead of class-level text templates')
    parser.add_argument('--prompt_mode', type=str, default='spatial+channel', choices=['spatial', 'channel', 'spatial+channel'])
    # idea 1: 'selective' uses LayerSelectNet single-forward per-sample layer decision,
    # 'fixed' falls back to the original args.stage fixed-layer injection
    parser.add_argument('--prompt_layer', type=str, default='selective', choices=['fixed', 'selective'])
    parser.add_argument('--decision_warm_bias', type=float, default=2.)
    parser.add_argument('--decision_lr_mult', type=float, default=0.02)
    parser.add_argument('--decision_warmup_epochs', type=int, default=10)
    parser.add_argument('--select_entropy_w', type=float, default=0.1)
    parser.add_argument('--no_template', action='store_true')
    parser.add_argument('--eqnorm', action='store_true', default=True)
    parser.add_argument('--stage', type=float, default=3.2, choices=[2, 2.1, 2.2, 2.3, 3, 3.1, 3.2, 3.3])
    # idea 3: 'bottleneck' adapter projector (default), 'linear' is the paper baseline
    parser.add_argument('--projector', type=str, default='bottleneck', choices=['linear', 'mlp', 'mlp3', 'bottleneck'])
    parser.add_argument('--bottleneck_dim', type=int, default=64)
    parser.add_argument('--align_weight', type=float, default=0.1,
                        help='InfoNCE weight pulling prompt embeddings toward visual prototypes; 0 disables')
    parser.add_argument('--preserve_weight', type=float, default=0.1,
                        help='weight keeping injected support features close to clean features; 0 disables')
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
