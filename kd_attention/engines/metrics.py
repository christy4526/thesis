'''
to torchtool?
'''
from __future__ import division

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from numbers import Number

from torchvision.models import resnet18
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from collections import defaultdict

import torch
from torch import distributed as dist
from ignite.metrics import Loss
from ignite.metrics import TopKCategoricalAccuracy
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric


class CallableMetric(Metric):
    def __init__(self, output_transform=lambda x: x):
        super().__init__(output_transform)

    def __call__(self, output):
        output = self._output_transform(output)
        self.update(output)


class MultiLoss(Loss, CallableMetric):
    def reset(self):
        self._sum = defaultdict(float)
        self._num_examples = defaultdict(int)

    def update(self, output):
        if len(output) == 2:
            y_pred, y = output
            kwargs = {}
        else:
            y_pred, y, kwargs = output
        average_losses = self._loss_fn(y_pred, y, **kwargs)

        for key, average_loss in average_losses.items():
            if len(average_loss.shape) != 0:
                raise ValueError('loss_fn did not return the average loss.')

            N = self._batch_size(y)
            self._sum[key] += average_loss.item() * N
            self._num_examples[key] += N

    def compute(self):
        average_loss = {}
        for key, num_examples in self._num_examples.items():
            if num_examples == 0:
                raise NotComputableError(
                    f'Loss "{key}" must have at least one example before it can be computed.')
            average_loss[key] = self._sum[key] / num_examples

        total = 0.
        for loss in average_loss.values():
            total += loss
        average_loss['total'] = total

        return average_loss


class DistributedMultiLoss(MultiLoss):
    def update(self, output):
        self.device = output[1].device
        super().update(output)

    def compute(self):
        average_loss = {}
        for key, _num_examples in self._num_examples.items():
            if _num_examples == 0:
                raise NotComputableError(
                    f'Loss "{key}" must have at least one example before it can be computed.')
            _sum = torch.as_tensor([self._sum[key]], device=self.device)
            _num_examples = torch.as_tensor(
                [_num_examples], device=self.device)

            dist.all_reduce(_sum)
            dist.all_reduce(_num_examples)

            average_loss[key] = _sum.item() / _num_examples.item()

        total = 0.
        for loss in average_loss.values():
            total += loss
        average_loss['total'] = total

        return average_loss


class DistributedLoss(Loss):
    def update(self, output):
        self.device = output[0].device
        super().update(output)

    def compute(self):
        _sum = torch.as_tensor([self._sum], device=self.device)
        _num_examples = torch.as_tensor(
            [self._num_examples], device=self.device
        )
        dist.all_reduce(_sum)
        dist.all_reduce(_num_examples)

        _sum = _sum.item()
        _num_examples = _num_examples.item()
        if _num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed.')
        return _sum / _num_examples


class DistributedTopKCategoricalAccuracy(TopKCategoricalAccuracy):
    def update(self, output):
        self.device = output[0].device
        super().update(output)

    def compute(self):
        _num_correct = torch.as_tensor(
            [self._num_correct], device=self.device
        )
        _num_examples = torch.as_tensor(
            [self._num_examples], device=self.device
        )
        dist.all_reduce(_num_correct)
        dist.all_reduce(_num_examples)

        _num_correct = _num_correct.item()
        _num_examples = _num_examples.item()

        if _num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed.')
        return _num_correct / _num_examples


class DistributedAccuracy(DistributedTopKCategoricalAccuracy):
    def __init__(self, output_transform=lambda x: x):
        super().__init__(k=1, output_transform=output_transform)


class MeanAUC(CallableMetric):
    def reset(self):
        self._sum = 0.
        self._num_examples = 0

    def _process(self, curves):
        if curves.ndim == 1:
            curves = curves.view(1, -1)
        xs = torch.linspace(0, 1, curves.size(1), device=curves.device)
        xs = xs.expand_as(curves)
        aucs = torch.trapz(curves, xs)
        return aucs

    def update(self, curves: torch.Tensor):
        aucs = self._process(curves)
        self._sum += aucs.sum().item()
        self._num_examples += aucs.size(0)

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed.')
        return self._sum / self._num_examples


class DistributedMeanAUC(MeanAUC):
    def update(self, output):
        self.device = output.device
        super().update(output)

    def compute(self):
        _sum = torch.as_tensor([self._sum], device=self.device)
        _num_examples = torch.as_tensor(
            [self._num_examples], device=self.device)

        dist.all_reduce(_sum)
        dist.all_reduce(_num_examples)

        _sum = _sum.item()
        _num_examples = _num_examples.item()

        if _num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed.')

        return _sum / _num_examples
