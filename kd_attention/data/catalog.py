from .imagenet import ImageNet
from .cub import CUB200
from .aircraft import FGVCAircraft
from .cars import Cars


def _get_just_category(annotation):
    return annotation['category']


class Catalog:
    num_classes = {
        'CUB200': 200,
        'Cars': 196,
        'FGVCAircraft-variant': 100,
        'FGVCAircraft-family': 70,
        'FGVCAircraft-manufacturer': 30,
        'ImageNet': 1000,
    }
    mean_std = {
        'CUB200': ([0.5016, 0.5161, 0.4459], [0.2193, 0.2132, 0.2594]),
        'Cars': ([0.4728, 0.4617, 0.4551], [0.2932, 0.2921, 0.3008]),
        'FGVCAircraft-variant': ([0.4855, 0.5209, 0.5488], [0.2152, 0.2060, 0.2378]),
        'FGVCAircraft-family': ([0.4855, 0.5209, 0.5488], [0.2152, 0.2060, 0.2378]),
        'FGVCAircraft-manufacturer': ([0.4855, 0.5209, 0.5488], [0.2152, 0.2060, 0.2378]),
        'ImageNet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    }
    cls_target_transforms = {
        'CUB200': _get_just_category,
        'Cars': _get_just_category,
        'FGVCAircraft-variant': _get_just_category,
        'FGVCAircraft-family': _get_just_category,
        'FGVCAircraft-manufacturer': _get_just_category,
        'ImageNet': None,
    }
    splits = {
        'CUB200': ['train', 'test'],
        'Cars': ['train', 'test'],
        'FGVCAircraft-variant': ['trainval', 'test'],
        'FGVCAircraft-family': ['trainval', 'test'],
        'FGVCAircraft-manufacturer': ['trainval', 'test'],
        'ImageNet': ['train', 'val'],
    }
    datasets_attrs = {
        'CUB200': (CUB200, {}),
        'Cars': (Cars, {}),
        'FGVCAircraft-variant': (FGVCAircraft, {'challenge': 'variant'}),
        'FGVCAircraft-family': (FGVCAircraft, {'challenge': 'family'}),
        'FGVCAircraft-manufacturer': (FGVCAircraft, {'challenge': 'manufacturer'}),
        'ImageNet': (ImageNet, {}),
    }
    rootdir_names = {
        'CUB200': 'CUB_200_2011',
        'Cars': 'Cars',
        'FGVCAircraft-variant': 'FGVC-Aircraft',
        'FGVCAircraft-family': 'FGVC-Aircraft',
        'FGVCAircraft-manufacturer': 'FGVC-Aircraft',
        'ImageNet': 'ILSVRC2017',
    }
