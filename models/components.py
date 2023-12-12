from typing import Callable, Optional

import torch
import torch.nn as nn
from timm.layers import DropPath


class MLP(nn.Sequential):
    def __init__(
      self,
      in_features: int,
      hidden_features: Optional[int] = None,
      out_features: Optional[int] = None,
      act_layer: Callable = nn.GELU,
      dropout_rate: float = 0.
    ) -> None:
        """2-layer MLP in Sequential

        :param in_features: number of input features
        :param hidden_features: number of hidden features, Default: None
        :param out_features: number of output features, Default: None
        :param act_layer: activation function, Default: 'GELU'
        :param dropout_rate: dropout rate, Default: 0
        """
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        dropout = nn.Dropout(dropout_rate)
        layers = nn.ModuleList([
          nn.Linear(in_features, hidden_features),
          act_layer(), dropout,
          nn.Linear(hidden_features, out_features), dropout
        ])
        super().__init__(*layers)


class Attention(nn.Module):
    def __init__(
      self, dim: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.
    ) -> None:
        """Multi-Head Self-Attention Module with Scale Dot Product Attention

        :param dim: embedding dimension
        :param num_heads: number of attention heads
        :param qkv_bias: whether to use bias in qkv projection, Default: True
        :param attn_drop: attention dropout rate, Default: 0
        :param proj_drop: projection dropout rate, Default: 0
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input feature map - tensor shape: (B, N, C), where:
            B is the batch size, N is the number of tokens, C is the embedding dimension
        :return: attention score - tensor shape: (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
        attn = q @ k.transpose(-2, -1)
        attn *= self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


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
