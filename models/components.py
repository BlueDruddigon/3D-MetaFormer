import itertools
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from utils.conv_utils import get_conv_layer
from utils.misc import ensure_tuple_dims


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


class PatchEmbed(nn.Module):
    def __init__(
      self,
      img_size: Union[int, Sequence[int]] = 224,
      patch_size: Union[int, Sequence[int]] = 16,
      spatial_dims: int = 2,
      in_chans: int = 3,
      embed_dim: int = 768,
      norm_layer: Optional[Callable] = None,
    ) -> None:
        """Patch Embedding with positional encodings based on spatial dimensions

        :param img_size: input image size, Default: 224
        :param patch_size: patch size, Default: 16
        :param spatial_dims: spatial dimensions, 2 for HW and 3 for DHW, Default: 2.
        :param in_chans: input channels, Default: 3.
        :param embed_dim: embedding dimension, Default: 768
        """
        super().__init__()
        
        img_size = ensure_tuple_dims(img_size, spatial_dims)
        patch_size = ensure_tuple_dims(patch_size, spatial_dims)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = np.prod([s // p for s, p in zip(img_size, patch_size)])
        self.patches_resolution = tuple(s // p for s, p in zip(img_size, patch_size))
        self.spatial_dims = spatial_dims
        
        # embeddings projection operator based on spatial_dims
        self.proj = get_conv_layer(
          spatial_dims,
          in_chans,
          embed_dim,
          kernel_size=self.patch_size,
          stride=self.patch_size,
          bias=True if norm_layer is None else False
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input image tensor - tensor shape: (B, C, [D], H, W), where
            B is batch size, C is channel dimension, and [D], H, W are spatial dimensions
        :return: a tensor contains patches with additional positional embeddings
        """
        assert x.size()[::-1][:self.spatial_dims] == self.img_size, "Input image size doesn't match model size"
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: Sequence[int], dim: int, norm_layer: Callable = nn.LayerNorm) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.spatial_dims = len(input_resolution)
        self.reduction = nn.Linear(dim * 2 ** self.spatial_dims, dim * 2, bias=False)
        self.norm = norm_layer(dim * 2 ** self.spatial_dims)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size()[1:-1] == self.input_resolution, 'input feature has wrong size'
        assert all(x % 2 == 0 for x in self.input_resolution), 'x size are not even'
        
        x = x.view([x.shape[0], *self.input_resolution, x.shape[-1]])
        if self.spatial_dims == 3:
            x = torch.cat(
              [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))],
              dim=-1,
            )
        elif self.spatial_dims == 2:
            x = torch.cat([x[:, i::2, j::2, :] for i, j in itertools.product(range(2), range(2))], dim=-1)
        else:
            raise ValueError
        
        x = self.reduction(self.norm(x))
        return x


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
        attn = attn * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class WindowAttention(nn.Module):
    def __init__(
      self,
      dim: int,
      spatial_dims: int,
      window_size: int,
      num_heads: int,
      qkv_bias: bool = True,
      attn_drop: float = 0.,
      proj_drop: float = 0.
    ) -> None:
        super().__init__()
        
        self.dim = dim
        self.window_size = ensure_tuple_dims(window_size, spatial_dims)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.spatial_dims = spatial_dims
        mesh_args = torch.meshgrid.__kwdefaults__
        
        window_range = (2*window_size - 1) ** spatial_dims
        self.relative_position_bias_table = nn.Parameter(torch.zeros(window_range, num_heads))
        if self.spatial_dims == 3:
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] = relative_coords[:, :, 0] + self.window_size[0] - 1
            relative_coords[:, :, 1] = relative_coords[:, :, 1] + self.window_size[1] - 1
            relative_coords[:, :, 2] = relative_coords[:, :, 2] + self.window_size[2] - 1
            relative_coords[:, :,
                            0] = relative_coords[:, :,
                                                 0] * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] = relative_coords[:, :, 1] * (2 * self.window_size[2] - 1)
        elif self.spatial_dims == 2:
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] = relative_coords[:, :, 0] + self.window_size[0] - 1
            relative_coords[:, :, 1] = relative_coords[:, :, 1] + self.window_size[1] - 1
            relative_coords[:, :, 0] = relative_coords[:, :, 0] * (2 * self.window_size[1] - 1)
        else:
            raise ValueError
        
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)]
        relative_position_bias = relative_position_bias.view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(B_ // num_windows, num_windows, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))
        
        return x


class UnetResBlock(nn.Module):
    def __init__(
      self,
      spatial_dims: int,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Sequence[int]] = 3,
      stride: Union[int, Sequence[int]] = 1,
      norm_layer: Callable = nn.BatchNorm3d,
      act_layer: Callable = nn.LeakyReLU,
      dropout_rate: float = 0.,
    ) -> None:
        super().__init__()
        
        self.block = UnetBasicBlock(
          spatial_dims,
          in_channels,
          out_channels,
          kernel_size=kernel_size,
          stride=stride,
          norm_layer=norm_layer,
          act_layer=act_layer,
          dropout_rate=dropout_rate
        )
        
        self.downsample = in_channels != out_channels
        if np.any(np.atleast_1d(stride) != 1):
            self.downsample = True
        
        if self.downsample:
            self.conv = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=1, stride=stride)
            self.drop = nn.Dropout(dropout_rate)
            self.norm = norm_layer(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for blk in list(self.block.children())[:-1]:
            x = blk(x)
        if self.downsample:
            residual = self.norm(self.drop(self.conv(residual)))
        x = x + residual
        x = self.block[-1](x)
        return x


class UnetBasicBlock(nn.Sequential):
    def __init__(
      self,
      spatial_dims: int,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Sequence[int]] = 3,
      stride: Union[int, Sequence[int]] = 1,
      norm_layer: Callable = nn.BatchNorm3d,
      act_layer: Callable = nn.LeakyReLU,
      dropout_rate: float = 0.,
    ) -> None:
        layers = nn.ModuleList([
          get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False),
          nn.Dropout(dropout_rate),
          norm_layer(out_channels),
          act_layer(),
          get_conv_layer(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False),
          nn.Dropout(dropout_rate),
          norm_layer(out_channels),
          act_layer(),
        ])
        super().__init__(*layers)
