import argparse
import os
import random
from typing import Optional, Union, Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timm.layers import to_ntuple
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
  model: Union[nn.Module, DDP],
  epoch: int,
  args: argparse.Namespace,
  best_acc: float = 0.,
  filename: str = '',
  optimizer: Optional[optim.Optimizer] = None,
  scheduler: Optional[LRScheduler] = None,
):
    """Save checkpoint utility

    :param model: Model to use, which is torch.nn.Module or DDP
    :param epoch: The current epoch
    :param args: The arguments that set.
    :param best_acc: Current best_valid_acc, Default: 0.
    :param filename: Filename to save, Default: ''
    :param optimizer: Optimizer to use, Default: None
    :param scheduler: LR Scheduler to use, Default: None
    """
    state_dict = model.module.state_dict() if args.distributed else model.state_dict()
    save_dict = {
      'state_dict': state_dict,
      'epoch': epoch,
      'best_valid_acc': best_acc,
    }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    if not filename:
        filename = f'weights-{epoch}-{best_acc:.2f}.pth'
    filepath = os.path.join(args.save_dir, filename)
    if args.rank == 0:
        torch.save(save_dict, filepath)
    print('Saved checkpoint', filepath)


def ensure_tuple_dims(tup: Any, dims: int) -> Sequence[int]:
    if isinstance(tup, (list, tuple)):
        assert len(tup) == dims
        return tuple(tup)
    elif isinstance(tup, int):
        return to_ntuple(dims)(tup)


def seed_everything(seed: int) -> None:
    """
    :param seed: seed to be set in random state
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class AverageMeter:
    """ Metrics tracking meter """
    def __init__(self) -> None:
        self.val = None
        self.avg = None
        self.count = None
        self.sum = None
        
        self.reset()
    
    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.count = 0
        self.sum = 0
    
    def update(self, value: float, n: int = 1) -> None:
        self.val = value
        self.count += n
        self.sum += value * n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)
