from torchvision.datasets.folder import default_loader
from torchvision.datasets import VisionDataset
import abc
import os
import pickle
import torch
from torchvision.datasets import ImageFolder
from typing import Callable
from typing import Union
from typing import Iterable
from numbers import Number
from copy import deepcopy


class ImageFolderTeacher(ImageFolder):
    def __init__(self, root, results_root, transform=None,
                 target_transform=None):
        super().__init__(root, transform, target_transform)
        self.results_root = results_root

    def _get_result_path(self, index):
        path, _ = self.imgs[index]
        f = f'{os.path.basename(path).split(".")[0]}.pkl'
        return os.path.join(self.results_root, f)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        result_path = self._get_result_path(index)
        with open(result_path, 'rb') as f:
            result = pickle.load(f)

        result = torch.tensor(result)

        return dict(image=sample, target=target, teacher_predict=result)


class MigratingDataset(VisionDataset, abc.ABC):
    ''' Abstract class for migration-needed-dataset that includes images and 
        imformation files.
    '''

    def __init__(self, root, transforms=None, transform=None,
                 target_transform=None):
        super().__init__(
            root, transforms=transforms, transform=transform,
            target_transform=target_transform
        )
        self.loader = default_loader
        self.samples = []
        self.classes = None
        self._migrate()
        assert self.classes is not None
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    @abc.abstractmethod
    def _migrate(self):
        return NotImplemented

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
