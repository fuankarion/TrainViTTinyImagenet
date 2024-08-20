# Adapted from
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch

from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)
  

class ViTv2(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, att_dim, 
                 depth, heads, mlp_dim, pool = 'cls', channels = 3, 
                 dropout = 0.1, emb_dropout = 0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, att_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, att_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, att_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(att_dim, 
                                       heads, 
                                       dim_feedforward=mlp_dim, 
                                       dropout=dropout,activation='gelu', 
                                       batch_first=True, 
                                       norm_first=True),
            depth
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(att_dim),
            nn.Linear(att_dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)