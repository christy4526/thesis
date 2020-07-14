from torch import distributed as dist
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .transforms import ToHalfTensor
from .catalog import Catalog


def build_transform(splits, mean=[0, 0, 0], std=[1, 1, 1], single_crop=True,
                    crop_size=224, resize_scale=(0.08, 1.0), half=True):
    if isinstance(splits, str):
        splits = [splits]
    transforms = {}
    for split in splits:
        if split in ('train', 'trainval'):
            transform = [
                T.RandomResizedCrop(crop_size, scale=resize_scale),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std)
            ]
        elif split in ('val', 'test'):
            if single_crop:
                crop = [
                    T.Resize(int(crop_size / 0.875)),
                    T.CenterCrop(crop_size),
                ]
            else:
                crop = [
                    T.Resize((crop_size, crop_size))
                ]
            transform = [
                *crop,
                T.ToTensor(),
                T.Normalize(mean, std)
            ]
        else:
            raise NotImplementedError(
                f'transform for split "{split}" is not implemented.')
        if half:
            transform.append(ToHalfTensor())
        transforms[split] = T.Compose(transform)
    return transforms


def build_target_transform(name):
    return Catalog.cls_target_transforms[name]


def build_datasets(name, root, splits=None, transforms={},
                   transform={}, target_transform={}):
    dataset_cls, extra_attrs = Catalog.datasets_attrs[name]
    if splits is None:
        splits = Catalog.splits[name]
    if isinstance(splits, str):
        splits = [splits]

    datasets = {}
    for split in splits:
        datasets[split] = dataset_cls(
            root,
            split,
            **extra_attrs,
            transforms=transforms.get(split, None),
            transform=transform.get(split, None),
            target_transform=target_transform.get(split, None)
        )

    return datasets


def build_dataloader(dataset, **kwargs):
    kwargs.setdefault('shuffle', None)
    kwargs.setdefault('sampler', None)
    kwargs.setdefault('batch_size', 1)

    sampler = kwargs.pop('sampler')
    shuffle = kwargs.pop('shuffle')
    batch_size = kwargs.pop('batch_size')

    if dist.is_initialized() and sampler is None:
        sampler = DistributedSampler(dataset)
        if batch_size > dist.get_world_size():
            batch_size = batch_size // dist.get_world_size()

    if shuffle is None:
        shuffle = sampler is None

    loader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size,
        shuffle=shuffle, **kwargs
    )
    return loader, sampler
