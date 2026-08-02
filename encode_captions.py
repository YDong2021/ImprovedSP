"""
Encode pre-generated captions with CLIP text encoder.
No text_length truncation — uses CLIP's full 77-token context.
Output: data/captions/{dataset}_{split}_clip_features.pt  [N_samples, 512]
"""
import os
import json
import argparse
import torch
import torch.nn.functional as F
import clip
from tqdm import tqdm


def main(args):
    # load captions
    cap_path = f'data/captions/{args.dataset}_{args.split}_captions.json'
    with open(cap_path, 'r', encoding='utf-8') as f:
        captions = json.load(f)
    n_samples = len(captions)
    print(f'Loaded {n_samples} captions from {cap_path}')

    # load CLIP (full context length, no truncation)
    model, _ = clip.load("ViT-B/32", device=f'cuda:{args.gpu}')
    model.eval()

    # encode in batches
    batch_size = args.batch_size
    features = torch.zeros(n_samples, 512)
    indices = sorted(captions.keys(), key=lambda x: int(x))

    for start in tqdm(range(0, n_samples, batch_size), desc='CLIP encoding'):
        batch_idx = indices[start:start + batch_size]
        texts = [captions[i] for i in batch_idx]
        tokens = clip.tokenize(texts, truncate=True).cuda(args.gpu)
        with torch.no_grad():
            feat = model.encode_text(tokens).float().cpu()
        for j, idx in enumerate(batch_idx):
            features[int(idx)] = feat[j]

    # eqnorm: normalize to uniform average norm
    avg_length = (features ** 2).sum(-1).sqrt().mean().item()
    features = F.normalize(features, dim=-1) * avg_length
    print(f'Avg feature norm before eqnorm: {avg_length:.2f}')

    out_path = f'data/captions/{args.dataset}_{args.split}_clip_features.pt'
    torch.save(features, out_path)
    print(f'Saved features {features.shape} to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    main(args)
