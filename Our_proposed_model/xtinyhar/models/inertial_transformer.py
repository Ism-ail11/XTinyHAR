
import torch
import torch.nn as nn
from torch import Tensor
from .blocks import PatchEmbed1D, TransformerEncoderLayer, PositionalEmbeddingLearned

class InertialTransformer(nn.Module):
    def __init__(self, in_channels=6, window_size=100, patch_size=20, dim=128, depth=2, heads=4, mlp_ratio=2.0, num_classes=27, drop=0.1):
        super().__init__()
        self.patch = PatchEmbed1D(in_channels, patch_size, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = PositionalEmbeddingLearned(seq_len=(window_size // patch_size) + 1, dim=dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim=dim, num_heads=heads, mlp_ratio=mlp_ratio, drop=drop, attn_drop=drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: Tensor):
        x = self.patch(x)  # (B, N, D)
        B, N, D = x.shape
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos(x)
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attn_maps.append(attn)
        x = self.norm(x)
        cls_tok = x[:, 0]
        logits = self.head(cls_tok)
        return logits, attn_maps
