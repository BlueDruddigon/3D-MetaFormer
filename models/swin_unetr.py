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
      qkv_bias: bool = True,
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
        self.num_layers = num_layers + 1  # plus 1 is the stage for the original input volumes
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
            if self.num_layers - i_layer == 1:
                # remove the residual block for the last encoded feature map
                # according to `https://github.com/Project-MONAI/MONAI/issues/4487`
                continue
            self.encoders.append(
              UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=in_chans if i_layer == 0 else embed_dim * 2 ** (i_layer - 1),
                out_channels=embed_dim if i_layer == 0 else embed_dim * 2 ** (i_layer - 1),
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True
              )
            )
        
        self.bottleneck = UnetrBasicBlock(
          spatial_dims,
          embed_dim * 2 ** num_layers,
          embed_dim * 2 ** num_layers,
          kernel_size=3,
          stride=1,
          norm_name=norm_name,
          res_block=True
        )
        
        self.out_proj = UnetOutBlock(spatial_dims, in_channels=embed_dim, out_channels=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.backbone(x)
        
        # Encode
        encoded_features = []
        for i in range(self.num_layers - 1):
            enc_out = self.encoders[i](x if i == 0 else hidden_states[i - 1])
            encoded_features.append(enc_out)
        # because of removing residual block for the last encoded feature map,
        # we just append the raw feature map to the encoded_features list
        encoded_features.append(hidden_states[-2])
        
        # Bottleneck
        dec_out = self.bottleneck(hidden_states[-1])
        
        # Decode
        for i in reversed(range(self.num_layers)):
            skip = encoded_features[i]
            dec_out = self.decoders[i](dec_out, skip)
        
        return self.out_proj(dec_out)
