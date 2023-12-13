from typing import Callable, Sequence, Union, Optional

import torch
import torch.nn as nn
from timm.layers import to_ntuple

from utils.conv_utils import get_conv_layer
from .backbones.vit import VisionTransformer


class UnetrBasicBlock(nn.Sequential):
    def __init__(
      self,
      spatial_dims: int,
      in_channels: int,
      out_channels: int,
      hidden_features: Optional[int] = None,
      kernel_size: Union[int, Sequence[int]] = 3,
      stride: Union[int, Sequence[int]] = 1,
      norm_layer: Callable = nn.BatchNorm3d,
      act_layer: Callable = nn.LeakyReLU,
      dropout_rate: float = 0.,
    ) -> None:
        hidden_features = hidden_features or out_channels
        layers = nn.ModuleList([
          get_conv_layer(
            spatial_dims, in_channels, hidden_features, kernel_size=kernel_size, stride=stride, bias=False
          ),
          nn.Dropout(dropout_rate),
          norm_layer(hidden_features),
          act_layer(),
          get_conv_layer(
            spatial_dims, hidden_features, out_channels, kernel_size=kernel_size, stride=stride, bias=False
          ),
          norm_layer(out_channels),
          act_layer(),
          nn.Dropout(dropout_rate),
        ])
        super().__init__(*layers)


class UnetrEncoderBlock(nn.Sequential):
    def __init__(
      self,
      spatial_dims: int,
      in_channels: int,
      out_channels: int,
      hidden_channels: Optional[int] = None,
      num_blocks: int = 1,
      kernel_size: Union[int, Sequence[int]] = 3,
      stride: Union[int, Sequence[int]] = 1,
      upsample_kernel_size: Union[int, Sequence[int]] = 2,
      norm_layer: Callable = nn.BatchNorm3d,
      act_layer: Callable = nn.LeakyReLU,
      dropout_rate: float = 0.
    ) -> None:
        hidden_channels = hidden_channels or out_channels
        layers = nn.ModuleList([
          get_conv_layer(
            spatial_dims,
            in_channels,
            hidden_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
            is_transposed=True
          ),
          UnetrBasicBlock(
            spatial_dims,
            hidden_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_layer=norm_layer,
            act_layer=act_layer,
            dropout_rate=dropout_rate,
          ),
        ])
        post_blocks = nn.ModuleList([
          nn.Sequential(
            get_conv_layer(
              spatial_dims,
              out_channels,
              out_channels,
              kernel_size=upsample_kernel_size,
              stride=upsample_kernel_size,
              is_transposed=True
            ),
            UnetrBasicBlock(
              spatial_dims,
              out_channels,
              out_channels,
              kernel_size=kernel_size,
              stride=stride,
              norm_layer=norm_layer,
              act_layer=act_layer,
              dropout_rate=dropout_rate
            )
          ) for _ in range(num_blocks - 1)  # minus 1 because we've formed the very first module before
        ])
        layers.extend(post_blocks)
        super().__init__(*layers)


class UnetrDecoderBlock(nn.Module):
    def __init__(
      self,
      spatial_dims: int,
      in_channels: int,
      out_channels: int,
      hidden_channels: Optional[int] = None,
      kernel_size: Union[int, Sequence[int]] = 3,
      stride: Union[int, Sequence[int]] = 1,
      upsample_kernel_size: Union[int, Sequence[int]] = 2,
      norm_layer: Callable = nn.BatchNorm3d,
      act_layer: Callable = nn.LeakyReLU,
      dropout_rate: float = 0.,
    ) -> None:
        super().__init__()
        
        hidden_channels = hidden_channels or out_channels
        self.deconv = get_conv_layer(
          spatial_dims,
          in_channels,
          hidden_channels,
          kernel_size=upsample_kernel_size,
          stride=upsample_kernel_size,
          is_transposed=True
        )
        
        self.conv_block = UnetrBasicBlock(
          spatial_dims,
          hidden_channels * 2,
          out_channels,
          kernel_size=kernel_size,
          stride=stride,
          norm_layer=norm_layer,
          act_layer=act_layer,
          dropout_rate=dropout_rate
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class UNETR(nn.Module):
    def __init__(
      self,
      in_chans: int,
      num_classes: int,
      img_size: int,
      feature_size: int = 64,
      embed_dim: int = 768,
      num_layers: int = 4,
      mlp_ratio: float = 4.,
      qkv_bias: bool = False,
      attn_drop_rate: float = 0.,
      proj_drop_rate: float = 0.,
      drop_path_rate: float = 0.1,
      num_heads: int = 12,
      norm_layer: Callable = nn.BatchNorm3d,
      act_layer: Callable = nn.ReLU6,
      use_checkpoint: bool = False,
      spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        
        self.num_layers = num_layers
        self.spatial_dims = spatial_dims
        self.img_size = to_ntuple(spatial_dims)(img_size)
        self.patch_size = to_ntuple(spatial_dims)(16)
        
        self.backbone = VisionTransformer(
          in_channels=in_chans,
          img_size=img_size,
          patch_size=16,
          spatial_dims=spatial_dims,
          embed_dim=embed_dim,
          depth=12,
          num_heads=num_heads,
          mlp_ratio=mlp_ratio,
          qkv_bias=qkv_bias,
          norm_layer=nn.LayerNorm,
          act_layer=nn.GELU,
          attn_drop=attn_drop_rate,
          proj_drop=proj_drop_rate,
          drop_path_rate=drop_path_rate,
          use_checkpoint=use_checkpoint,
          backbone_only=True,
        )
        self.input_resolution = self.backbone.input_resolution
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(num_layers):
            enc_layer = UnetrBasicBlock(
              spatial_dims,
              in_chans,
              feature_size,
              kernel_size=3,
              stride=1,
              norm_layer=norm_layer,
              act_layer=act_layer,
              dropout_rate=proj_drop_rate
            ) if i == 0 else UnetrEncoderBlock(
              spatial_dims,
              embed_dim,
              feature_size * 2 ** i,
              num_blocks=num_layers - i,
              kernel_size=3,
              stride=1,
              upsample_kernel_size=2,
              norm_layer=norm_layer,
              act_layer=act_layer,
              dropout_rate=proj_drop_rate
            )
            self.encoders.append(enc_layer)
            self.decoders.append(
              UnetrDecoderBlock(
                spatial_dims,
                in_channels=feature_size * 2 ** (i + 1) if i < num_layers - 1 else embed_dim,
                out_channels=feature_size * 2 ** i,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_layer=norm_layer,
                act_layer=act_layer,
                dropout_rate=proj_drop_rate
              )
            )
        
        self.out_proj = get_conv_layer(spatial_dims, feature_size, num_classes, kernel_size=1)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.input_resolution) + [embed_dim]
    
    def proj_feat(self, x):
        new_view = [x.shape[0]] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.backbone(x, intermediate_levels=[3, 6, 9, 12])
        
        # Encode
        encoded_features = []
        for i in range(self.num_layers):
            if i == 0:
                enc_out = self.encoders[i](x)
            else:
                enc_out = self.encoders[i](self.proj_feat(hidden_states[i - 1]))
            encoded_features.append(enc_out)
        
        # Decode
        dec_out = None
        for i in reversed(range(self.num_layers)):
            skip = encoded_features[i]
            if i == self.num_layers - 1:
                out = self.decoders[i](self.proj_feat(hidden_states[-1]), skip=skip)
            else:
                out = self.decoders[i](dec_out, skip=skip)
            dec_out = out
        
        return self.out_proj(dec_out)
