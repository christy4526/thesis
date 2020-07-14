import os
from typing import Callable
from scipy import io

from .datasets import MigratingDataset


class NotCompatibleAnnotationFileError(Exception):
    pass


class Cars(MigratingDataset):
    '''Pytorch compatible dataset with Cars dataset from Stanford AI laboratory
    Initialization:
        root (str): path of directory that contains images.
        split (str): split indicator. choises: ["train", "test"]
        trasnforms (callable): simultaneous transform method.
        trasnform (callable): image transform method.
        target_transform (callable): target transform method.

    Note:
        the class indices of Cars dataset starts with 1 originally, because of
        the compatibility with Matlab (I think).
        it is changed to starts with 0, because of the compatibility with
        Pytorch.
        the right structure of the dataset is:
        root
            - train
                - 00000.jpg
                - 00001.jpg
                ...
            - test
                - 00000.jpg
                - 00001.jpg
                ...
            - devkit
                - cars_meta.mat
                - cars_test_annos.mat
                ...
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
        if split == 'train':
            annfile = 'cars_train_annos.mat'
        else:
            annfile = 'cars_test_annos_withlabels.mat'
        self.annfile = os.path.join(root, 'devkit', annfile)
        self.metafile = os.path.join(root, 'devkit', 'cars_meta.mat')

        super().__init__(root, transforms, transform, target_transform)

    def _migrate(self):
        annotations = io.loadmat(self.annfile, squeeze_me=True)['annotations']
        for i, annotation in enumerate(annotations):
            if len(annotation) != 6:
                raise NotCompatibleAnnotationFileError(
                    f'file "{os.path.basename(self.annfile)}" has not'
                    'enough annotation informations. '
                    'maybe not "cars_test_anno.mat" '
                    'but "cars_test_annos_withlabels.mat"?'
                )
            xmin, ymin, xmax, ymax, clslabel, fname = annotation
            annotation = {
                'category': clslabel - 1,
                'xmin': xmin, 'ymin': ymin,
                'xmax': xmax, 'ymax': ymax,
            }
            self.samples.append(
                (os.path.join(self.root, self.split, fname), annotation)
            )

        meta = io.loadmat(self.metafile, squeeze_me=True)
        self.classes = meta['class_names']

    def _extra_repr(self):
        return f'split: {self.split}'
    extra_repr = _extra_repr
