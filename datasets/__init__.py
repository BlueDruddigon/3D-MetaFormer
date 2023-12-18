import argparse

from monai.data.thread_buffer import ThreadDataLoader

from .btcv import AbdomenDataset
from .transformations import get_default_transforms


def build_dataset(args: argparse.Namespace):
    transforms = get_default_transforms(args)
    train_set = AbdomenDataset(
      args.data_root,
      transform=transforms,
      is_train=True,
      train_cache_num=args.train_cache_num,
      num_workers=args.workers
    )
    valid_set = AbdomenDataset(
      args.data_root,
      transform=transforms,
      is_train=False,
      valid_cache_num=args.valid_cache_num,
      num_workers=args.workers
    )
    
    # data sampler and dataloader
    train_loader = ThreadDataLoader(
      train_set,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.workers,
      sampler=None,
      pin_memory=True,
      persistent_workers=True
    )
    
    valid_loader = ThreadDataLoader(
      valid_set,
      batch_size=1,
      shuffle=False,
      num_workers=args.workers,
      sampler=None,
      pin_memory=True,
      persistent_workers=True
    )
    
    return train_loader, valid_loader
