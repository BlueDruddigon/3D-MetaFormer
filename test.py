import torch

from models.swin_unetr import SwinUNETR

swin = SwinUNETR(1, 1, (96, 96, 96), embed_dim=48)
print(swin)
