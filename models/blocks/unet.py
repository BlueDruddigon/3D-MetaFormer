from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from utils.conv_utils import get_conv_layer


class UnetBasicBlock(nn.Sequential):
    def __init__(
      self,
      spatial_dims: int,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Sequence[int]] = 3,
      stride: Union[int, Sequence[int]] = 1,
      norm_layer: Callable[..., nn.Module] = nn.InstanceNorm3d,
      act_layer: Callable[..., nn.Module] = nn.LeakyReLU,
      dropout_rate: Optional[float] = None
    ) -> None:
        layers = nn.ModuleList([
          get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size, stride=stride, bias=False),
          nn.Dropout(dropout_rate) if dropout_rate is not None else nn.Identity(),
          norm_layer(out_channels),
          act_layer(),
          get_conv_layer(spatial_dims, out_channels, out_channels, kernel_size, stride=1, bias=False),
          nn.Dropout(dropout_rate) if dropout_rate is not None else nn.Identity(),
          norm_layer(out_channels),
          act_layer(),
        ])
        super().__init__(*layers)


class UnetResBlock(nn.Module):
    def __init__(
      self,
      spatial_dims: int,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Sequence[int]] = 3,
      stride: Union[int, Sequence[int]] = 1,
      norm_layer: Callable[..., nn.Module] = nn.InstanceNorm3d,
      act_layer: Callable[..., nn.Module] = nn.LeakyReLU,
      dropout_rate: Optional[float] = None
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
            self.drop = nn.Dropout(dropout_rate) if dropout_rate is not None else nn.Identity()
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
