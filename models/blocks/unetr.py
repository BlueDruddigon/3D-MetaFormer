from typing import Callable, Sequence, Union

import torch.nn as nn

from .unet import UnetBasicBlock, UnetResBlock


class UnetrBasicBlock(nn.Sequential):
    def __init__(
      self,
      spatial_dims: int,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Sequence[int]] = 3,
      stride: Union[int, Sequence[int]] = 1,
      norm_layer: Callable[..., nn.Module] = nn.InstanceNorm3d,
      act_layer: Callable[..., nn.Module] = nn.LeakyReLU,
      dropout_rate: float = 0.,
      is_residual: bool = False
    ) -> None:
        block = UnetResBlock if is_residual else UnetBasicBlock
        super().__init__(
          block(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_layer=norm_layer,
            act_layer=act_layer,
            dropout_rate=dropout_rate
          )
        )
