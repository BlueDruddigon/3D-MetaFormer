from typing import Callable, Optional, Sequence, Set, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_ntuple, trunc_normal_
from torch.utils.checkpoint import checkpoint

from ..components import MLP, PatchEmbed, PatchMerging, WindowAttention


def pad(dims: Sequence[int], window_size: int) -> Sequence[int]:
    spatial_dims = len(dims)
    if spatial_dims == 3:
        D, H, W = dims
        pad_d = (window_size - D%window_size) % window_size
        pad_r = (window_size - W%window_size) % window_size
        pad_b = (window_size - H%window_size) % window_size
        return D + pad_d, H + pad_b, W + pad_r
    elif spatial_dims == 2:
        H, W = dims
        pad_r = (window_size - W%window_size) % window_size
        pad_b = (window_size - H%window_size) % window_size
        return H + pad_b, W + pad_r
    raise ValueError


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    x_shape = x.size()
    if len(x_shape) == 5:
        B, D, H, W, C = x.shape
        x = x.view(B, D // window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    elif len(x_shape) == 4:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    else:
        raise ValueError
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, dims: Sequence[int]) -> torch.Tensor:
    spatial_dims = len(dims)
    B = int(windows.shape[0] / (torch.prod(torch.tensor(dims)) / window_size ** spatial_dims))
    if spatial_dims == 3:
        D, H, W = dims
        x = windows.view(
          B, D // window_size, H // window_size, W // window_size, window_size, window_size, window_size, -1
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    elif spatial_dims == 2:
        H, W = dims
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    else:
        raise ValueError
    return x


def compute_mask(dims: Sequence[int], window_size: int, shift_size: int):
    cnt = 0
    spatial_dims = len(dims)
    dims = pad(dims, window_size)
    img_mask = torch.zeros((1, *dims, 1))
    if spatial_dims == 3:
        for d in slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None):
            for h in slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None):
                for w in slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt = cnt + 1
    elif spatial_dims == 2:
        for h in slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None):
            for w in slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None):
                img_mask[:, h, w, :] = cnt
                cnt = cnt + 1
    else:
        raise ValueError
    
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size ** spatial_dims)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    
    return attn_mask


class SwinTransformerBlock(nn.Module):
    def __init__(
      self,
      dim: int,
      input_resolution: Sequence[int],
      num_heads: int,
      window_size: int = 7,
      shift_size: int = 0,
      mlp_ratio: float = 4.,
      qkv_bias: bool = True,
      attn_drop: float = 0.,
      proj_drop: float = 0.,
      drop_path: float = 0.,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm
    ) -> None:
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.spatial_dims = len(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size <= self.window_size, 'shift_size must be between 0 and window_size'
        
        self.attn = WindowAttention(
          dim,
          spatial_dims=self.spatial_dims,
          window_size=window_size,
          num_heads=num_heads,
          qkv_bias=qkv_bias,
          attn_drop=attn_drop,
          proj_drop=proj_drop
        )
        self.attn_norm = norm_layer(dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = MLP(dim, mlp_hidden_dim, act_layer=act_layer, dropout_rate=proj_drop)
        self.ffn_norm = norm_layer(dim)
        
        attn_mask = None
        if self.shift_size > 0:
            attn_mask = compute_mask(self.input_resolution, self.window_size, self.shift_size)
        self.register_buffer('attn_mask', attn_mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size()[1:-1] == self.input_resolution, 'input feature has wrong size'
        C = x.shape[-1]
        
        shortcut = x
        x = self.attn_norm(x)
        x = x.view([x.shape[0], *self.input_resolution, C])
        
        # pad feature maps to multiples of window size
        new_resolution = pad(self.input_resolution, self.window_size)
        if len(new_resolution) == 3:
            D, H, W = self.input_resolution
            x = F.pad(x, (0, 0, 0, new_resolution[2] - W, 0, new_resolution[1] - H, 0, new_resolution[0] - D))
        elif len(new_resolution) == 2:
            H, W = self.input_resolution
            x = F.pad(x, (0, 0, 0, new_resolution[1] - W, 0, new_resolution[0] - H))
        
        # cyclic shifts
        if self.shift_size > 0:
            dims = tuple(range(1, self.spatial_dims + 1))
            shifted_x = torch.roll(x, shifts=to_ntuple(self.spatial_dims)(-self.shift_size), dims=dims)
        else:
            shifted_x = x
        
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size ** self.spatial_dims, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # merge windows
        attn_windows = attn_windows.view([-1, *to_ntuple(self.spatial_dims)(self.window_size), C])
        shifted_x = window_reverse(attn_windows, self.window_size, new_resolution)
        
        # reverse cyclic shifts
        if self.shift_size > 0:
            dims = tuple(range(1, self.spatial_dims + 1))
            x = torch.roll(shifted_x, shifts=to_ntuple(self.spatial_dims)(self.shift_size), dims=dims)
        else:
            x = shifted_x
        
        # un-pad features
        if len(new_resolution) == 3:
            D, H, W = self.input_resolution
            x = x[:, :D, :H, :W, :].contiguous()
        elif len(new_resolution) == 2:
            H, W = self.input_resolution
            x = x[:, :H, :W, :].contiguous()
        
        x = shortcut + self.drop_path(x)
        
        # FFN
        x = x + self.drop_path(self.ffn(self.ffn_norm(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(
      self,
      dim: int,
      input_resolution: Sequence[int],
      depth: int,
      num_heads: int,
      window_size: int,
      mlp_ratio: float = 4.,
      qkv_bias: bool = True,
      attn_drop: float = 0.,
      proj_drop: float = 0.,
      drop_path: Union[float, Sequence[float]] = 0.,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      downsample: Optional[Callable[..., nn.Module]] = None,
      use_checkpoint: bool = False
    ) -> None:
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.spatial_dims = len(input_resolution)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # build blocks
        self.blocks = nn.ModuleList([
          SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0 if i % 2 == 0 else window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer
          ) for i in range(depth)
        ])
        
        # patch merging layer
        self.downsample = downsample(
          input_resolution, dim=dim, norm_layer=norm_layer
        ) if downsample is not None else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, *range(2, self.spatial_dims + 2), 1)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        return x.permute(0, self.spatial_dims + 1, *range(1, self.spatial_dims + 1))


class SwinTransformer(nn.Module):
    def __init__(
      self,
      img_size: Union[int, Sequence[int]] = 224,
      patch_size: Union[int, Sequence[int]] = 4,
      in_chans: int = 3,
      num_classes: int = 1000,
      embed_dim: int = 96,
      depths: Sequence[int] = (2, 2, 6, 2),
      num_heads: Sequence[int] = (3, 6, 12, 24),
      window_size: int = 7,
      mlp_ratio: float = 4.,
      qkv_bias: bool = True,
      attn_drop_rate: float = 0.,
      proj_drop_rate: float = 0.,
      drop_path_rate: float = 0.1,
      norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      ape: bool = False,
      patch_norm: bool = False,
      use_checkpoint: bool = False,
      spatial_dims: int = 2,
      backbone_only: bool = False
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.spatial_dims = spatial_dims
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** self.num_layers)
        
        # patch embeddings within positional encodings
        self.patch_embed = PatchEmbed(
          img_size=img_size,
          patch_size=patch_size,
          spatial_dims=spatial_dims,
          in_chans=in_chans,
          embed_dim=embed_dim,
          norm_layer=norm_layer if patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        self.input_resolution = self.patch_embed.patches_resolution
        
        # absolute position embeddings
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        
        self.pos_drop = nn.Dropout(proj_drop_rate)
        
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
              dim=int(embed_dim * 2 ** i_layer),
              input_resolution=tuple(s // (2 ** i_layer) for s in self.input_resolution),
              depth=depths[i_layer],
              num_heads=num_heads[i_layer],
              window_size=window_size,
              mlp_ratio=mlp_ratio,
              qkv_bias=qkv_bias,
              attn_drop=attn_drop_rate,
              proj_drop=proj_drop_rate,
              drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
              norm_layer=norm_layer,
              act_layer=act_layer,
              downsample=PatchMerging,  # here also using PatchMerging layer at the end of feature extraction
              use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
        
        self.classification = not backbone_only
        if self.classification:
            self.norm = norm_layer(self.num_features)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x: torch.Tensor, save_state: bool = False) -> torch.Tensor:
        hidden_states_out = []
        
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = x.contiguous().view(x.shape[0], self.embed_dim, *self.input_resolution)
        hidden_states_out.append(x)
        
        for layer in self.layers:
            x = layer(x)
            hidden_states_out.append(x)
        
        x = x.permute(0, *range(2, self.spatial_dims + 2), 1)
        return hidden_states_out if save_state else x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.classification:
            x = self.forward_features(x, save_state=False)
            x = self.norm(x)
            x = self.pool(x.transpose(1, 2))
            x = torch.flatten(x, 1)
            return self.head(x)
        return self.forward_features(x, save_state=True)
    
    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return {'absolute_pos_embed'}
