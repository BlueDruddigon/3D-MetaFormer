from typing import Callable, Optional, Union, Sequence

import torch
import torch.nn as nn
from timm.layers import to_ntuple, trunc_normal_
from torch.utils.checkpoint import checkpoint

from models.components import VitBlock
from utils.conv_utils import get_conv_layer


class PatchEmbed(nn.Module):
    def __init__(
      self,
      img_size: int = 224,
      patch_size: int = 16,
      spatial_dims: int = 2,
      in_chans: int = 3,
      embed_dim: int = 768,
      dropout_rate: float = 0.
    ) -> None:
        super(PatchEmbed, self).__init__()
        
        self.img_size = to_ntuple(spatial_dims)(img_size)
        self.patch_size = to_ntuple(spatial_dims)(patch_size)
        self.num_patches = (img_size // patch_size) ** spatial_dims
        self.patches_resolution = to_ntuple(spatial_dims)(img_size // patch_size)
        self.spatial_dims = spatial_dims
        
        # embeddings
        self.proj = get_conv_layer(
          spatial_dims, in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout_rate)
        
        trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, C, [D], H, W) - input image tensor
        :return: a tensor contains patches with additional positional embeddings
        """
        assert x.size()[::-1][:self.spatial_dims] == self.img_size, "Input image size doesn't match model size"
        x = self.proj(x).flatten(2).transpose(1, 2)
        x += self.pos_embed
        return self.pos_drop(x)


class VisionTransformer(nn.Module):
    def __init__(
      self,
      img_size: int = 224,
      patch_size: int = 16,
      spatial_dims: int = 2,
      in_channels: int = 3,
      num_classes: int = 1000,
      embed_dim: int = 768,
      representation_size: Optional[int] = None,
      depth: int = 12,
      num_heads: int = 12,
      mlp_ratio: float = 4.,
      qkv_bias: bool = False,
      norm_layer: Callable = nn.LayerNorm,
      act_layer: Callable = nn.GELU,
      attn_drop: float = 0.,
      proj_drop: float = 0.,
      drop_path_rate: float = 0.,
      use_checkpoint: bool = False,
      backbone_only: bool = False
    ) -> None:
        super(VisionTransformer, self).__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        
        self.patch_embed = PatchEmbed(
          img_size=img_size,
          patch_size=patch_size,
          spatial_dims=spatial_dims,
          in_chans=in_channels,
          embed_dim=embed_dim,
          dropout_rate=proj_drop
        )
        self.num_patches = self.patch_embed.num_patches
        self.input_resolution = self.patch_embed.patches_resolution
        
        # Stochastic Depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # building blocks
        self.blocks = nn.ModuleList([
          VitBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=dpr[i]
          ) for i in range(depth)
        ])
        
        self.norm_layer = norm_layer(embed_dim)
        self.classification = not backbone_only
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, std=.02)
            self.head = nn.Linear(embed_dim, num_classes)
            if representation_size is not None:
                self.head = nn.Sequential(
                  nn.Linear(embed_dim, representation_size),
                  nn.Tanh(),
                  nn.Linear(representation_size, num_classes),
                )
        
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    def forward_features(
      self,
      x: torch.Tensor,
      save_state: bool = False,
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        x = self.patch_embed(x)
        # cls_token adding
        if self.classification:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        
        hidden_state_out = []
        for func in self.blocks:  # forward encoder block using checkpointing
            if self.use_checkpoint:
                x = checkpoint(func, x)
            else:
                x = func(x)
            hidden_state_out.append(x)
        
        x = self.norm_layer(x)
        return hidden_state_out if save_state else x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.forward_features(x, save_state=False)
        if self.classification:
            x = self.head(x[:, 0])  # classifier `token` as used by standard language architecture
        return x
