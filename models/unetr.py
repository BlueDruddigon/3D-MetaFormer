from typing import Sequence

import torch
import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from timm.layers import to_ntuple

from .backbones.vit import VisionTransformer


class UNETR(nn.Module):
    def __init__(
      self,
      in_chans: int,
      num_classes: int,
      img_size: Sequence[int],
      feature_size: int = 64,
      embed_dim: int = 768,
      num_layers: int = 4,
      mlp_ratio: float = 4.,
      qkv_bias: bool = False,
      attn_drop_rate: float = 0.,
      proj_drop_rate: float = 0.,
      drop_path_rate: float = 0.1,
      num_heads: int = 12,
      norm_name: str = 'instance',
      use_checkpoint: bool = False,
      spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        
        assert len(img_size) == spatial_dims
        
        self.num_layers = num_layers
        self.spatial_dims = spatial_dims
        self.img_size = img_size
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
              spatial_dims, in_chans, feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=True
            ) if i == 0 else UnetrPrUpBlock(
              spatial_dims,
              embed_dim,
              feature_size * 2 ** i,
              num_layer=num_layers - i - 1,
              kernel_size=3,
              stride=1,
              upsample_kernel_size=2,
              norm_name=norm_name,
              conv_block=True,
              res_block=False
            )
            self.encoders.append(enc_layer)
            self.decoders.append(
              UnetrUpBlock(
                spatial_dims,
                in_channels=feature_size * 2 ** (i + 1) if i < num_layers - 1 else embed_dim,
                out_channels=feature_size * 2 ** i,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=False
              )
            )
        
        self.out_proj = UnetOutBlock(spatial_dims, in_channels=feature_size, out_channels=num_classes)
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
