import argparse

from monai.data import CacheDataset, DataLoader, load_decathlon_datalist

from .samplers import CustomSampler
from .transformations import get_default_transforms


def build_dataset(args: argparse.Namespace):
    transforms = get_default_transforms(args)
    train_files = load_decathlon_datalist('./datasets/data.json', True, 'training')
    valid_files = load_decathlon_datalist('./datasets/data.json', True, 'validation')
    train_ds = CacheDataset(
      train_files,
      transform=transforms['train'],
      cache_num=args.train_cache_num,
      cache_rate=1.0,
      num_workers=args.workers
    )
    valid_ds = CacheDataset(
      valid_files,
      transform=transforms['valid'],
      cache_num=args.valid_cache_num,
      cache_rate=1.0,
      num_workers=args.workers
    )
    
    # data sampler and dataloader
    train_sampler = CustomSampler(train_ds) if args.distributed else None
    train_loader = DataLoader(
      train_ds,
      batch_size=args.batch_size,
      shuffle=train_sampler is None,
      num_workers=args.workers,
      sampler=train_sampler,
      pin_memory=True,
      persistent_workers=True
    )
    
    valid_sampler = CustomSampler(valid_ds, shuffle=False) if args.distributed else None
    valid_loader = DataLoader(
      valid_ds,
      batch_size=1,
      shuffle=False,
      num_workers=args.workers,
      sampler=valid_sampler,
      pin_memory=True,
      persistent_workers=True
    )
    
    return train_loader, valid_loader
