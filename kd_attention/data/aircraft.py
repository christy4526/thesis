import os
from typing import Callable

from collections import OrderedDict as odict

from torchvision.datasets.folder import pil_loader
import numpy as np

from .datasets import MigratingDataset


class FGVCAircraft(MigratingDataset):
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
                 challenge: str = 'manufacturer',
                 transforms: Callable = None,
                 transform: Callable = None,
                 target_transform: Callable = None):
        if not split in ['test', 'train', 'trainval', 'val']:
            raise ValueError(
                f'choose `test` or `train` or `trainval` or `val` not `{split}`'
            )
        if not challenge in ['variant', 'family', 'manufacturer']:
            raise ValueError(
                f'choose `variant` or `family`, `manufacturer` not `{split}`'
            )
        self.split = split
        self.challenge = challenge
        if not root.endswith('data') or not root.endswith('data/'):
            root = os.path.join(root, 'data')
        super().__init__(root, transforms, transform, target_transform)
        self.loader = self._remove_copyright

    def _load_annotation(self):
        fname = os.path.join(
            self.root, f'images_{self.challenge}_{self.split}.txt'
        )

        data = odict()
        f = open(fname, 'r')
        for line in f:
            line = line.split()
            imageid = line[0]
            clsname = ' '.join(line[1:])
            data[imageid] = clsname
        f.close()
        return data

    def _load_bbox(self):
        bboxfile = 'images_box.txt'
        _bboxes = np.loadtxt(os.path.join(self.root, bboxfile), dtype=str)
        bboxes = odict()
        coordnames = ['xmin', 'ymin', 'xmax', 'ymax']
        for _bbox in _bboxes:
            bboxes[_bbox[0]] = {
                key: float(_bbox[i])-1
                for i, key in enumerate(coordnames, 1)
            }
        return bboxes

    def _load_classes(self):
        if self.challenge == 'family':
            clsnamefile = 'families.txt'
        else:
            clsnamefile = f'{self.challenge}s.txt'
        f = open(os.path.join(self.root, clsnamefile), 'r')
        clsnames = [line[:-1] for line in f.readlines()]
        f.close()
        return clsnames

    def _migrate(self):
        ann = self._load_annotation()
        bbox = self._load_bbox()
        self.classes = self._load_classes()

        images_dir = os.path.join(self.root, 'images')
        for imgid in ann.keys():
            annotation = {
                'category': self.classes.index(ann[imgid]),
            }
            annotation.update(bbox[imgid])
            self.samples.append(
                (os.path.join(images_dir, f'{imgid}.jpg'), annotation)
            )

    def _remove_copyright(self, path):
        img = pil_loader(path)
        width, height = img.size
        return img.crop((0, 0, width, height-20))

    def _extra_repr(self):
        split = f'split: {self.split}'
        challenge = f'challenge: {self.challenge}'
        return '\n'.join([split, challenge])
    extra_repr = _extra_repr
