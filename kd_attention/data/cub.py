import os
from typing import Callable
import numpy as np

from .datasets import MigratingDataset


class CUB200(MigratingDataset):
    '''Pytorch compatible dataset with CUB200 dataset from Caltech
    Initialization:
        root (str): path of directory that contains images.
        split (str): split indicator. choises: ["train", "test"]
        trasnforms (callable): simultaneous transform method.
        trasnform (callable): image transform method.
        target_transform (callable): target transform method.
    '''

    def __init__(self,
                 root: str,
                 split: str = 'train',
                 transforms: Callable = None,
                 transform: Callable = None,
                 target_transform: Callable = None):
        if not split in ['test', 'train']:
            raise ValueError(f'choose `test` or `train`, not `{split}`')
        self.split = split
        super().__init__(root, transforms, transform, target_transform)

    def _migrate(self):
        fname_info = np.loadtxt(
            os.path.join(self.root, 'images.txt'),
            dtype=str
        )
        split_info = np.loadtxt(
            os.path.join(self.root, 'train_test_split.txt'),
            dtype=int
        )
        bbox_annotation = np.loadtxt(
            os.path.join(self.root, 'bounding_boxes.txt'),
            dtype=float
        )
        cls_info = np.loadtxt(
            os.path.join(self.root, 'image_class_labels.txt'),
            dtype=int
        )
        cns = np.loadtxt(
            os.path.join(self.root, 'classes.txt'),
            dtype=str,
            usecols=(1,)
        )
        self.classes = [name.split('.')[1] for name in cns]
        for a, b, c, d in zip(fname_info, split_info, bbox_annotation, cls_info):
            a, b, c, d, = list(map(lambda x: int(x[0]), [a, b, c, d]))
            # image id integrity
            assert a == b == c == d

        iterator = zip(
            fname_info[:, 1],
            split_info[:, 1],
            bbox_annotation[:, 1:],
            cls_info[:, 1]
        )
        images_dir = os.path.join(self.root, 'images')
        for fname, split, bbox, class_idx in iterator:
            if split != ['test', 'train'].index(self.split):
                continue
            bbox = {k: v for k, v in zip(['x', 'y', 'width', 'height'], bbox)}
            annotation = {
                'category': class_idx-1
            }
            annotation.update(bbox)
            self.samples.append((os.path.join(images_dir, fname), annotation))

    def _extra_repr(self):
        return f'split: {self.split}'
    extra_repr = _extra_repr
