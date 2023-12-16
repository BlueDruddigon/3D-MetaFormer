from typing import Callable, Sequence, Union

import torch
import torch.nn as nn

from utils.conv_utils import get_conv_layer
from .backbones.swin import SwinTransformer
from .unetr import UnetrBasicBlock


class SwinEncoderBlock(nn.Sequential):
    def __init__(
      self,
      spatial_dims: int,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Sequence[int]] = 3,
      stride: Union[int, Sequence[int]] = 1,
      norm_layer: Callable = nn.InstanceNorm3d,
      act_layer: Callable = nn.LeakyReLU,
      dropout_rate: float = 0.,
      is_residual: bool = True
    ) -> None:
        layer = UnetrBasicBlock(
          spatial_dims,
          in_channels,
          out_channels,
          kernel_size=kernel_size,
          stride=stride,
          norm_layer=norm_layer,
          act_layer=act_layer,
          dropout_rate=dropout_rate,
          is_residual=is_residual
        )
        super(SwinEncoderBlock, self).__init__(layer)


class SwinDecoderBlock(nn.Module):
    def __init__(
      self,
      spatial_dims: int,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Sequence[int]] = 3,
      stride: Union[int, Sequence[int]] = 1,
      upsample_kernel_size: Union[int, Sequence[int]] = 2,
      norm_layer: Callable = nn.BatchNorm3d,
      act_layer: Callable = nn.LeakyReLU,
      dropout_rate: float = 0.,
      is_residual: bool = True
    ) -> None:
        super(SwinDecoderBlock, self).__init__()
        
        self.deconv = get_conv_layer(
          spatial_dims,
          in_channels,
          out_channels,
          kernel_size=upsample_kernel_size,
          stride=upsample_kernel_size,
          is_transposed=True
        )
        self.conv_block = UnetrBasicBlock(
          spatial_dims,
          out_channels * 2,
          out_channels,
          kernel_size=kernel_size,
          stride=stride,
          norm_layer=norm_layer,
          act_layer=act_layer,
          dropout_rate=dropout_rate,
          is_residual=is_residual
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class SwinUNETR(nn.Module):
    patch_size: int = 2
    window_size: int = 7
    
    def __init__(
      self,
      in_chans: int,
      num_classes: int,
      img_size: Sequence[int],
      depths: Sequence[int] = (2, 2, 2, 2),
      num_heads: Sequence[int] = (3, 6, 12, 24),
      embed_dim: int = 64,
      mlp_ratio: float = 4.,
      qkv_bias: bool = False,
      attn_drop_rate: float = 0.,
      proj_drop_rate: float = 0.,
      drop_path_rate: float = 0.1,
      norm_layer: Callable = nn.InstanceNorm3d,
      act_layer: Callable = nn.ReLU6,
      patch_norm: bool = False,
      use_checkpoint: bool = False,
      spatial_dims: int = 3
    ) -> None:
        super(SwinUNETR, self).__init__()
        
        assert len(img_size) == spatial_dims
        
        num_layers = len(depths)
        self.num_layers = num_layers + 1
        self.spatial_dims = spatial_dims
        
        self.backbone = SwinTransformer(
          img_size=img_size,
          patch_size=self.patch_size,
          in_chans=in_chans,
          embed_dim=embed_dim,
          depths=depths,
          num_heads=num_heads,
          window_size=self.window_size,
          mlp_ratio=mlp_ratio,
          qkv_bias=qkv_bias,
          attn_drop_rate=attn_drop_rate,
          proj_drop_rate=proj_drop_rate,
          drop_path_rate=drop_path_rate,
          norm_layer=nn.LayerNorm,
          act_layer=nn.GELU,
          ape=False,
          patch_norm=patch_norm,
          use_checkpoint=use_checkpoint,
          spatial_dims=spatial_dims,
          backbone_only=True
        )
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            self.encoders.append(
              SwinEncoderBlock(
                spatial_dims,
                in_channels=in_chans if i_layer == 0 else embed_dim * 2 ** (i_layer - 1),
                out_channels=embed_dim if i_layer == 0 else embed_dim * 2 ** (i_layer - 1),
                kernel_size=3,
                stride=1,
                norm_layer=norm_layer,
                act_layer=act_layer,
                dropout_rate=proj_drop_rate,
                is_residual=True
              )
            )
            self.decoders.append(
              SwinDecoderBlock(
                spatial_dims,
                in_channels=embed_dim * 2 ** i_layer,
                out_channels=embed_dim if i_layer == 0 else embed_dim * 2 ** (i_layer - 1),
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_layer=norm_layer,
                act_layer=act_layer,
                dropout_rate=proj_drop_rate,
                is_residual=True
              )
            )
        
        self.bottleneck = UnetrBasicBlock(
          spatial_dims,
          embed_dim * 2 ** num_layers,
          embed_dim * 2 ** num_layers,
          kernel_size=3,
          stride=1,
          norm_layer=norm_layer,
          act_layer=act_layer,
          dropout_rate=proj_drop_rate,
          is_residual=True
        )
        
        self.out_proj = get_conv_layer(spatial_dims, embed_dim, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.backbone(x)
        
        # Encode
        encoded_features = []
        for i in range(self.num_layers):
            if i == 0:
                enc_out = self.encoders[i](x)
            else:
                enc_out = self.encoders[i](hidden_states[i - 1])
            encoded_features.append(enc_out)
        
        # Bottleneck
        dec_out = self.bottleneck(hidden_states[-1])
        
        # Decode
        for i in reversed(range(self.num_layers)):
            skip = encoded_features[i]
            dec_out = self.decoders[i](dec_out, skip)
        
        return self.out_proj(dec_out)
