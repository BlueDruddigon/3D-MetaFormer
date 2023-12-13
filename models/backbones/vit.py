from typing import Callable, Optional, Union, Sequence

import torch
import torch.nn as nn
from timm.layers import trunc_normal_, DropPath
from torch.utils.checkpoint import checkpoint

from ..components import MLP, Attention, PatchEmbed


class VitBlock(nn.Module):
    def __init__(
      self,
      dim: int,
      num_heads: int,
      qkv_bias: bool = True,
      mlp_ratio: float = 4.,
      act_layer: Callable = nn.GELU,
      norm_layer: Callable = nn.LayerNorm,
      attn_drop: float = 0.,
      proj_drop: float = 0.,
      drop_path: float = 0.,
    ) -> None:
        """Vision Transformer Encoder Block

        :param dim:
        :param num_heads: Number of attention heads
        :param qkv_bias: Whether to add a bias to the Attention module, Default: True
        :param mlp_ratio: The ratio of mlp dimensions, Default: 4.
        :param act_layer: The activation function, Default: 'GELU'
        :param norm_layer: The normalization layer, Default: 'LayerNorm'
        :param attn_drop: The dropout for attention, Default: 0.
        :param proj_drop: The dropout for projection, Default: 0.
        :param drop_path: The Stochastic Depth decay ratio, Default: 0.
        """
        super().__init__()
        
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.attn_norm = norm_layer(dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.ffn = MLP(dim, mlp_hidden_dim, act_layer=act_layer, dropout_rate=proj_drop)
        self.ffn_norm = norm_layer(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input feature map - tensor shape: (B, N, C), where
            B is the batch size, N is the number of tokens, and C is the number of embedding dimensions
        :return: encoded feature map - tensor shape: (B, N, C)
        """
        x += self.drop_path(self.attn(self.attn_norm(x)))
        x += self.drop_path(self.ffn(self.ffn_norm(x)))
        return x


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
        """Modified version of Vision Transformer Architecture

        :param img_size: input image size, Default: 224
        :param patch_size: patch size, Default: 16
        :param spatial_dims: spatial dimensions, Default: 2
        :param in_channels: number of input channels, Default: 3
        :param num_classes: number of classes for classification, Default: 1000
        :param embed_dim: embedding dimension, Default: 768
        :param representation_size: representation size for classifier, Default: None
        :param depth: depth of the encoder, Default: 12
        :param num_heads: number of attention heads, Default: 12
        :param mlp_ratio: the ratio of mlp hidden dimensions, Default: 4.
        :param qkv_bias: whether to add bias for the qkv heads in Attention, Default: False
        :param norm_layer: normalization layer, Default: 'LayerNorm'
        :param act_layer: activation layer, Default: 'GELU'
        :param attn_drop: attention dropout rate, Default: 0.
        :param proj_drop: projection dropout rate, Default: 0.
        :param drop_path_rate: stochastic depth rate, Default: 0.1
        :param use_checkpoint: whether to use checkpointing for fast training, Default: False
        :param backbone_only: whether to return the backbone only, Default: False
        """
        super().__init__()
        
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
        num_patches = self.patch_embed.num_patches
        self.input_resolution = self.patch_embed.patches_resolution
        
        # absolute position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(proj_drop)
        
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
        trunc_normal_(self.pos_embed, std=.02)
    
    @staticmethod
    def _init_weights(m):
        """Weight initialization for specified modules"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        """Make torch.jit happy"""
        return {'pos_embed', 'cls_token'}
    
    def forward_features(
      self,
      x: torch.Tensor,
      save_state: bool = False,
      intermediate_levels: Optional[Sequence[int]] = None
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """feature extractor forward pass

        :param x: input image tensor - tensor shape (B, C, [D], H, W) where
            B is batch size, C is a channel dimension, [D], H, W is spatial dimensions
        :param save_state: whether to return all hidden states instead of last output, Default: False
        :param intermediate_levels: list of intermediate feature levels to save, Default: None
        :return: list of hidden states if save_state is True, else last hidden state only
        """
        x = self.patch_embed(x)
        x += self.pos_embed  # positional encodings
        x = self.pos_drop(x)
        # cls_token adding
        if self.classification:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        
        hidden_state_out = []
        for idx, func in enumerate(self.blocks):  # forward encoder block using checkpointing
            if self.use_checkpoint:
                x = checkpoint(func, x)
            else:
                x = func(x)
            if not save_state:
                continue
            if intermediate_levels is None:  # if `intermediate_levels` is not provided, save all states
                hidden_state_out.append(x)
                continue
            elif idx + 1 in intermediate_levels:  # add 1 because of start index = 0
                hidden_state_out.append(x)
        
        x = self.norm_layer(x)
        return hidden_state_out if save_state else x
    
    def forward(self, x: torch.Tensor, intermediate_levels: Optional[Sequence[int]] = None) -> torch.Tensor:
        """
        :param x: input image tensor - shape (B, C, [D], H, W) where
            B is batch size, C is a channel dimension, [D], H, W is spatial dimensions
        :param intermediate_levels: list of intermediate feature levels to save, Default: None
        :return: extracted feature maps or classification heads
        """
        if self.classification:
            out = self.forward_features(x, save_state=False)
            out = self.head(out[:, 0])  # classifier `token` as used by standard language architecture
            return out
        return self.forward_features(x, save_state=True, intermediate_levels=intermediate_levels)
