import os
from torch import distributed as dist

__all__ = [
    'get_world_size',
    'synchronize',
    'initialize_distributed',
    'get_rank',
]


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        if 'WORLD_SIZE' in os.environ:
            return int(os.environ['WORLD_SIZE'])
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        if 'RANK' in os.environ:
            return int(os.environ['RANK'])
        return 0
    return dist.get_rank()


def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    if get_world_size() == 1:
        return
    dist.barrier()


def initialize_distributed():
    if get_world_size() == 1:
        return False

    dist.init_process_group('nccl', init_method='env://')
    synchronize()

    return True


def remove_strs(name, olds):
    for old in olds:
        name = name.replace(old, '')
    return name


def make_explanation_name(image_path, suffix=''):
    name = os.path.basename(image_path).split('.')[0]
    if suffix:
        name = f'{name}_{suffix}.pth'
    else:
        name = f'{name}.pth'
    return name
