from typing import Sequence

import torch
import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

from .backbones.swin import SwinTransformer


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
      norm_name: str = 'instance',
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
        
        self.decoders = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            self.decoders.append(
              UnetrUpBlock(
                spatial_dims,
                in_channels=embed_dim * 2 ** i_layer,
                out_channels=embed_dim if i_layer == 0 else embed_dim * 2 ** (i_layer - 1),
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True
              )
            )
        
        self.in_proj = UnetrBasicBlock(
          spatial_dims, in_chans, embed_dim, kernel_size=3, stride=1, norm_name=norm_name, res_block=True
        )
        self.out_proj = UnetOutBlock(spatial_dims, in_channels=embed_dim, out_channels=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.backbone(x)
        
        input_features = self.in_proj(x)
        encoded_features = [input_features, *hidden_states]
        
        # Decode
        dec_out = encoded_features[-1]
        for i in reversed(range(self.num_layers)):
            skip = encoded_features[i]
            dec_out = self.decoders[i](dec_out, skip)
        
        return self.out_proj(dec_out)
