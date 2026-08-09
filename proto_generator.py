# MFGN-style transformer decoder for prototype generation.
# Reference: Yu et al., Masked Feature Generation Network for Few-Shot Learning, IJCAI 2022.
# Instead of reconstructing sample features, this decoder reconstructs the prototypes
# of other disjoint subsets of the same class from SP-conditioned seed support features.

import torch
import torch.nn as nn

from weight_init import trunc_normal_


class GenBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # x: [L, N, dim]
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class PrototypeGenerator(nn.Module):
    """MFGN-style decoder: generates candidate prototypes from seed features.

    The input sequence contains one seed token (a seed support feature) and
    num_gen mask tokens. Each position j is distinguished by a learned sample
    number embedding, which enables one-to-many diverse generation.
    The outputs at the mask token positions are the generated prototype candidates.
    """
    def __init__(self, dim=384, num_heads=6, depth=2, num_gen=5, mlp_ratio=4.,
                 drop=0., attn_drop=0.):
        super().__init__()
        self.num_gen = num_gen
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.num_embed = nn.Embedding(num_gen + 1, dim)
        self.blocks = nn.ModuleList([
            GenBlock(dim, num_heads, mlp_ratio, drop, attn_drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.pred = nn.Linear(dim, dim)

        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.num_embed.weight, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, seed):
        # seed: [N, dim]
        seed_tok = seed.unsqueeze(0) + self.num_embed.weight[0].view(1, 1, -1)  # [1, N, dim]
        mask_tok = self.mask_token + self.num_embed.weight[1:self.num_gen + 1].view(self.num_gen, 1, -1)
        mask_tok = mask_tok.repeat(1, seed.shape[0], 1)                         # [M, N, dim]
        x = torch.cat([seed_tok, mask_tok], dim=0)                              # [M+1, N, dim]
        for b in self.blocks:
            x = b(x)
        x = self.pred(self.norm(x[1:]))                                         # [M, N, dim]
        return x.permute(1, 0, 2).contiguous()                                  # [N, M, dim]
