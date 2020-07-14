import torch
from PIL import Image


class ToHalfTensor(object):
    def __call__(self, tensor):
        return tensor.half()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Denormalize:
    def __init__(self, mean=[0., 0., 0., ], std=[1., 1., 1., ], inplace=True):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, x):
        return denormalize(x, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class MinMaxNormalize:
    def __init__(self, inplace=True, nanto=0., infto=0.):
        self.inplace = inplace
        self.nanto = nanto
        self.infto = infto

    def __call__(self, x):

        return min_max_normalize(x, self.inplace, self.nanto, self.infto)

    def __repr__(self):
        return self.__class__.__name__ + '()'

# functional


def denormalize(tensor, mean, std, inplace=False):
    squeeze = False
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        squeeze = True
    assert tensor.ndim == 4
    if not inplace:
        tensor = tensor.clone()
    mean = torch.as_tensor(mean, device=tensor.device)
    std = torch.as_tensor(std, device=tensor.device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    tensor.mul_(std).add_(mean)
    if squeeze:
        tensor = tensor.squeeze(0)
    return tensor


def min_max_normalize(tensor, inplace=False, nanto=0., infto=0.):
    if not inplace:
        tensor = tensor.clone()
    minima = tensor.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    tensor.sub_(minima)
    maxima = tensor.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    tensor.div_(maxima)

    tensor = torch.where(
        torch.isnan(tensor),
        torch.zeros_like(tensor).fill_(nanto),
        tensor
    )
    tensor = torch.where(
        torch.isinf(tensor),
        torch.zeros_like(tensor).fill_(infto),
        tensor
    )

    return tensor


class MultiInputWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inputs):
        '''
        # generally
        inputs = iter(inputs)
        image, others = next(inputs), list(inputs)
        # python >= 3.0
        image, *others = inputs
        # yes generally
        '''
        image, others = inputs[0], inputs[1:]
        image = self.transform(image)
        return image, others

    def __repr__(self):
        return f'{self.__class__.__name__}(transform={self.transform})'


class DetachFileName:
    def __call__(self, image):
        return image, image.fname

    def __repr__(self):
        return self.__class__.__name__ + '()'


def preserving_fname_loader(path):
    image = Image.open(path)
    fname = image.fp.name
    converted = image.convert('RGB')
    converted.fname = fname
    return converted
