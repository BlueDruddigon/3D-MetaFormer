from typing import Union, Sequence, Optional, Tuple

import numpy as np
from monai.networks.layers import Conv


def get_conv_layer(
  spatial_dims: int,
  in_channels: int,
  out_channels: int,
  kernel_size: Union[int, Sequence[int]] = 3,
  stride: Union[int, Sequence[int]] = 1,
  padding: Optional[Union[int, Sequence[int]]] = None,
  output_padding: Optional[Union[int, Sequence[int]]] = None,
  dilation: Union[int, Sequence[int]] = 1,
  groups: int = 1,
  bias: bool = False,
  is_transposed: bool = False,
) -> Union[Conv.CONV, Conv.CONVTRANS]:
    if padding is None:
        padding = get_padding(kernel_size, stride)
    conv_type = Conv[Conv.CONVTRANS if is_transposed else Conv.CONV, spatial_dims]
    
    if is_transposed:
        if output_padding is None:
            output_padding = get_output_padding(kernel_size, stride, padding)
        conv = conv_type(
          in_channels,
          out_channels,
          kernel_size,
          stride=stride,
          padding=padding,
          output_padding=output_padding,
          groups=groups,
          bias=bias,
          dilation=dilation
        )
    else:
        conv = conv_type(
          in_channels,
          out_channels,
          kernel_size,
          stride=stride,
          padding=padding,
          dilation=dilation,
          groups=groups,
          bias=bias
        )
    
    return conv


def get_padding(
  kernel_size: Union[int, Sequence[int]],
  stride: Union[int, Sequence[int]],
) -> Union[int, Tuple[int, ...]]:
    kernel_size = np.atleast_1d(kernel_size)
    stride = np.atleast_1d(stride)
    padding = (kernel_size-stride+1) / 2
    if np.min(padding) < 0:
        raise AssertionError
    padding = tuple(int(p) for p in padding)
    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
  kernel_size: Union[int, Sequence[int]], stride: Union[int, Sequence[int]], padding: Union[int, Tuple[int]]
) -> Union[int, Tuple[int, ...]]:
    kernel_size = np.atleast_1d(kernel_size)
    stride = np.atleast_1d(stride)
    padding = np.atleast_1d(padding)
    
    output_padding = 2*padding + stride - kernel_size
    if np.min(output_padding) < 0:
        raise AssertionError
    output_padding = tuple(int(p) for p in output_padding)
    
    return output_padding if len(output_padding) > 1 else output_padding[0]
