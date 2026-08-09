# Stage-2 training: MFGN-style prototype generation on top of the SP model.
# The SP encoder (student + t2i/t2i2/se_block) is frozen and kept in eval mode
# (BN protection). Only the PrototypeGenerator decoder is trained.
# Supervision: prototypes of disjoint same-class subsets are reconstructed from
# SP-conditioned seed support features (targets detached), combined with the
# meta classification loss on the augmented prototypes.

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
from proto_generator import PrototypeGenerator
from data.dataloader import EpisodeSampler
from data.dataset import DatasetWithTextLabel
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
    test_aug = transforms.Compose([transforms.Resize(int(args.image_size * 1.1)),
                                   transforms.CenterCrop(args.image_size),
                                   transforms.ToTensor(),
                                   norm])

    # per episode, each class needs (num_gen + 1) disjoint support subsets plus 15 queries
    n_sub = args.num_gen + 1
    n_per = args.shot * n_sub + 15

    train_dataset = DatasetWithTextLabel(args.dataset, train_aug, split='train')
    n_episodes = args.train_episodes
    args.train_way = args.way if args.train_way == -1 else args.train_way
    if n_episodes == -1:
        n_episodes = int(len(train_dataset) / (args.train_way * n_per))
    episode_sampler = EpisodeSampler(train_dataset.dataset.targets,
                                     n_episodes,
                                     args.train_way,
                                     n_per, fix_seed=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=episode_sampler, num_workers=8)
    num_classes = len(train_dataset.dataset.classes)

    test_dataset = DatasetWithTextLabel(args.dataset, test_aug, split=args.split)
    episode_sampler = EpisodeSampler(test_dataset.dataset.targets, args.episodes, args.way, n_per)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=episode_sampler, num_workers=0)

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

    # build the SP student model with the same module layout as train_vit_sp.py
    if args.model == 'visformer-t':
        student = visformer.visformer_tiny(num_classes=num_classes)
    elif args.model == 'visformer-t-84':
        student = visformer.visformer_tiny_84(num_classes=num_classes)
    else:
        raise ValueError(f'unknown model: {args.model}')

    args.feature_dim = 384
    if 2 <= args.stage < 3:
        args.feature_dim = 192
    if args.projector == 'linear':
        student.t2i = torch.nn.Linear(text_dim, args.feature_dim, bias=False)
    elif args.projector == 'mlp':
        student.t2i = torch.nn.Sequential(torch.nn.Linear(text_dim, text_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(text_dim, args.feature_dim, bias=False))
    elif args.projector == 'mlp3':
        student.t2i = torch.nn.Sequential(torch.nn.Linear(text_dim, text_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(text_dim, text_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(text_dim, args.feature_dim, bias=False))
    if 'channel' in args.prompt_mode:
        student.t2i2 = torch.nn.Linear(text_dim, args.feature_dim, bias=False)
        student.se_block = torch.nn.Sequential(torch.nn.Linear(args.feature_dim * 2, args.feature_dim, bias=True),
                                               torch.nn.Sigmoid(),
                                               torch.nn.Linear(args.feature_dim, args.feature_dim),
                                               torch.nn.Sigmoid(),)

    # load the SP checkpoint and freeze the whole student (BN protection: keep eval)
    checkpoint = torch.load(args.init, map_location=f'cuda:{args.gpu}')
    missing, unexpected = student.load_state_dict(checkpoint['state_dict'], strict=False)
    if missing or unexpected:
        raise ValueError(f'SP checkpoint mismatch: missing={missing}, unexpected={unexpected}')
    student = student.cuda(args.gpu)
    student.requires_grad_(False)
    student.eval()

    # build the prototype generator (the only trainable module)
    generator = PrototypeGenerator(dim=args.feature_dim, num_heads=args.gen_heads, depth=args.gen_depth,
                                   num_gen=args.num_gen)
    generator = generator.cuda(args.gpu)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu}')
        generator.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print(f'load generator checkpoint at epoch {start_epoch}')

    optim = torch.optim.AdamW(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.test:
        test(test_text, student, generator, test_loader, 0, args)
        return

    best_acc = 0.
    for epoch in range(start_epoch, args.epochs):
        train(train_text, student, generator, train_loader, optim, epoch, args)

        if (epoch + 1) % args.test_freq == 0:
            acc = test(test_text, student, generator, test_loader, epoch, args)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': generator.state_dict(),
            'optimizer': optim.state_dict(),
        }
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


def extract_features(image, glabels, text, student, args):
    """SP-conditioned feature extraction for support images (no gradient)."""
    text_features = text[glabels]
    if args.prompt_mode == 'spatial':
        text_features = student.t2i(text_features)
        _, sup_im_features = student.forward_with_semantic_prompt(image, text_features, args)
    else:
        _, sup_im_features = student.forward_with_semantic_prompt_channel(image, text_features, args)
    return sup_im_features


def train(text, student, generator, train_loader, optim, epoch, args):
    student.eval()  # BN protection: the frozen encoder stays in eval mode
    generator.train()
    n_sub = args.num_gen + 1
    n_sup = args.shot * n_sub
    losses, losses_mse, accs = 0., 0., 0.
    for idx, episode in enumerate(train_loader):
        image = episode[0].cuda(args.gpu)  # way * n_per
        glabels = episode[1].cuda(args.gpu)
        labels = torch.arange(args.train_way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)

        image = image.view(args.train_way, n_sup + 15, *image.shape[1:])
        sup, que = image[:, :n_sup].contiguous(), image[:, n_sup:].contiguous()
        sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

        glabels = glabels.view(args.train_way, n_sup + 15)[:, :n_sup].contiguous().view(-1)

        # the frozen encoder needs no gradient
        with torch.no_grad():
            sup_im_features = extract_features(sup, glabels, text, student, args)
            # subset 0 is the seed subset; subsets 1..num_gen provide the reconstruction targets
            sup_im_features = sup_im_features.view(args.train_way, n_sub, args.shot, -1)
            proto_a = sup_im_features[:, 0].mean(dim=1)             # seed subset prototype
            proto_b = sup_im_features[:, 1:].mean(dim=2)            # target subset prototypes
            seed_feats = sup_im_features[:, 0].reshape(-1, args.feature_dim)
            _, que_im_features = student(que)

        # generate candidate prototypes from each seed support feature, then average over shots
        candidates = generator(seed_feats)                                        # way*shot x M x dim
        candidates = candidates.view(args.train_way, args.shot, args.num_gen, -1).mean(dim=1)

        l_mse = F.mse_loss(candidates, proto_b)
        proto_final = torch.cat([proto_a.unsqueeze(1), candidates], dim=1).mean(dim=1)
        sim = F.normalize(que_im_features, dim=-1) @ F.normalize(proto_final, dim=-1).t()
        l_meta = F.cross_entropy(sim / args.t, labels)
        loss = l_meta + args.lamb * l_mse
        losses += loss.item()
        losses_mse += l_mse.item()
        _, pred = sim.max(-1)
        accs += labels.eq(pred).sum().float().item() / labels.shape[0]

        optim.zero_grad()
        loss.backward()
        optim.step()

        if idx % args.print_step == 0 or idx == len(train_loader) - 1:
            print_string = (f'Train epoch: {epoch}, step: {idx:3d}, loss: {losses / (idx + 1):.4f}, '
                            f'mse: {losses_mse / (idx + 1):.4f}, acc: {accs * 100 / (idx + 1):.2f}')
            print(print_string)
    args.logger.add_scalar('train/loss', losses / len(train_loader), epoch)
    args.logger.add_scalar('train/acc', accs / len(train_loader), epoch)


def test(text, student, generator, test_loader, epoch, args):
    student.eval()
    generator.eval()
    n_sub = args.num_gen + 1
    n_sup = args.shot * n_sub
    accs, accs_base = [], []
    with torch.no_grad():
        for episode in test_loader:
            image = episode[0].cuda(args.gpu)  # way * n_per
            glabels = episode[1].cuda(args.gpu)
            labels = torch.arange(args.way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)

            image = image.view(args.way, n_sup + 15, *image.shape[1:])
            sup, que = image[:, :n_sup].contiguous(), image[:, n_sup:].contiguous()
            sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

            glabels = glabels.view(args.way, n_sup + 15)[:, :n_sup].contiguous().view(-1)

            sup_im_features = extract_features(sup, glabels, text, student, args)
            sup_im_features = sup_im_features.view(args.way, n_sub, args.shot, -1)
            proto_a = sup_im_features[:, 0].mean(dim=1)
            seed_feats = sup_im_features[:, 0].reshape(-1, args.feature_dim)
            _, que_im_features = student(que)

            candidates = generator(seed_feats)
            candidates = candidates.view(args.way, args.shot, args.num_gen, -1).mean(dim=1)

            if args.test_classifier == 'prototype':
                proto_final = torch.cat([proto_a.unsqueeze(1), candidates], dim=1).mean(dim=1)
                sim = F.normalize(que_im_features, dim=-1) @ F.normalize(proto_final, dim=-1).t()
                _, pred = sim.max(-1)
                sim_base = F.normalize(que_im_features, dim=-1) @ F.normalize(proto_a, dim=-1).t()
                _, pred_base = sim_base.max(-1)
            elif args.test_classifier == 'fc':
                from sklearn.linear_model import LogisticRegression
                # treat generated candidates as extra support features of each class
                x_train = torch.cat([seed_feats.view(args.way, args.shot, -1), candidates], dim=1)
                x_train = F.normalize(x_train, dim=-1).view(-1, args.feature_dim).cpu().numpy()
                y_train = torch.arange(args.way).unsqueeze(-1).repeat(1, args.shot + args.num_gen).view(-1).numpy()
                x_test = que_im_features.cpu().numpy()
                clf = LogisticRegression(penalty='l2',
                                         random_state=0,
                                         C=1,
                                         solver='lbfgs',
                                         max_iter=1000,
                                         multi_class='multinomial')
                clf.fit(x_train, y_train)
                pred = torch.tensor(clf.predict(x_test)).cuda(args.gpu)
                # baseline: seed support features only
                x_train_base = F.normalize(seed_feats, dim=-1).cpu().numpy()
                y_train_base = torch.arange(args.way).unsqueeze(-1).repeat(1, args.shot).view(-1).numpy()
                clf.fit(x_train_base, y_train_base)
                pred_base = torch.tensor(clf.predict(x_test)).cuda(args.gpu)

            accs.append(labels.eq(pred).sum().float().item() / labels.shape[0])
            accs_base.append(labels.eq(pred_base).sum().float().item() / labels.shape[0])

    m, h = mean_confidence_interval(accs)
    m_base, h_base = mean_confidence_interval(accs_base)
    print(f'Test epoch: {epoch}, SP baseline acc: {m_base * 100:.2f}+-{h_base * 100:.2f}, '
          f'proto-gen acc: {m * 100:.2f}+-{h * 100:.2f}')
    args.logger.add_scalar('test/acc', m * 100, epoch)
    args.logger.add_scalar('test/acc_baseline', m_base * 100, epoch)

    return m


if __name__ == '__main__':
    # use 'spawn' instead of the default 'fork' to avoid segfault when
    # multiprocessing DataLoader workers start in this container environment
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='proto_gen')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100'])
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--image_size', type=int, default=224, choices=[224, 84])
    parser.add_argument('--aug', action='store_true', default=True)
    parser.add_argument('--model', type=str, default='visformer-t', choices=['visformer-t', 'visformer-t-84'])
    parser.add_argument('--nlp_model', type=str, default='clip', choices=['clip', 'glove', 'mpnet'])
    parser.add_argument('--prompt_mode', type=str, default='spatial+channel', choices=['spatial', 'channel', 'spatial+channel'])
    parser.add_argument('--no_template', action='store_true')
    parser.add_argument('--eqnorm', action='store_true', default=True)
    parser.add_argument('--stage', type=float, default=3.2, choices=[2, 2.1, 2.2, 2.3, 3, 3.1, 3.2, 3.3])
    parser.add_argument('--projector', type=str, default='linear', choices=['linear', 'mlp', 'mlp3'])
    parser.add_argument('--avg', type=str, default='all', choices=['all', 'patch', 'head'])
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--init', type=str, default='checkpoint/miniImageNet/visformer-t/sp/checkpoint_epoch_best.pth')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--text_length', type=int, default=20)
    parser.add_argument('--train_way', type=int, default=-1)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--train_episodes', type=int, default=-1)
    parser.add_argument('--episodes', type=int, default=600)
    parser.add_argument('--test_classifier', type=str, default='prototype', choices=['prototype', 'fc'])
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=10)
    # prototype generator settings
    parser.add_argument('--num_gen', type=int, default=5)
    parser.add_argument('--gen_depth', type=int, default=2)
    parser.add_argument('--gen_heads', type=int, default=6)
    parser.add_argument('--lamb', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)

    args = parser.parse_args()
    if args.seed >= 0:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    main(args)
