import random

import numpy as np
import torch


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
