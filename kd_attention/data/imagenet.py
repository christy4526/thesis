from torchvision.datasets import ImageFolder
import os
from torchvision.datasets import ImageNet as _ImageNet
import numpy as np


class ImageNet(ImageFolder):
    '''for compatability with the older version
    '''

    def __init__(self, root, split='train', transforms=None,
                 transform=None, target_transform=None):
        _root = os.path.join(root, 'Data', 'CLS-LOC', split)
        super().__init__(_root, transform, target_transform)
        self.winds = self.classes

        map_clsloc = np.loadtxt(
            os.path.join(root, 'devkit', 'data', 'map_clsloc.txt'),
            usecols=(0, 2),
            dtype=str
        )
        self.wind_to_class = dict(map_clsloc)
        self.class_to_wind = {v: k for k, v in self.wind_to_class.items()}
        self.classes = [self.wind_to_class[wind] for wind in self.winds]
        self.class_to_idx = {k: self.winds.index(self.class_to_wind[k])
                             for k in self.classes}
