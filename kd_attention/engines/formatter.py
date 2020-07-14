import torch
from copy import deepcopy

from ignite.metrics import Accuracy

from .metrics import Loss, DistributedLoss
from .metrics import MultiLoss, DistributedMultiLoss
from .metrics import MeanAUC, DistributedMeanAUC
from .metrics import DistributedAccuracy
from .metrics import DistributedTopKCategoricalAccuracy
from .metrics import TopKCategoricalAccuracy


def _grabber(name):
    def grabber(x):
        x, y = x[0][name], x[1]
        if x.dtype == torch.half:
            x = x.float()
        return x, y
    return grabber


def classification_metrics(criterion, distributed=False):
    if distributed:
        loss = DistributedMultiLoss(criterion)
        acc = DistributedAccuracy()
        acc5 = DistributedTopKCategoricalAccuracy()
    else:
        loss = MultiLoss(criterion)
        acc = Accuracy()
        acc5 = TopKCategoricalAccuracy()

    metrics = {
        'loss': loss,
        'top1acc': acc,
        'top5acc': acc5
    }
    return metrics


def classification_metric_formatter(metrics):
    metrics = deepcopy(metrics)
    loss = metrics.pop('loss')
    fmtstr = 'loss: {:.4f}, '.format(loss['celoss'])
    fmtstr += 'top1 acc: {top1acc:.4f}, '
    fmtstr += 'top5 acc: {top5acc:.4f}, '

    return fmtstr.format(**metrics), loss, metrics


def kd_metrics(criterion, distributed=False):
    if distributed:
        loss = DistributedMultiLoss(criterion)
        acc_student = DistributedAccuracy(
            _grabber('output_student'))
        acc5_student = DistributedTopKCategoricalAccuracy(
            5, _grabber('output_student'))
    else:
        loss = MultiLoss(criterion)
        acc_student = Accuracy(
            _grabber('output_student'))
        acc5_student = TopKCategoricalAccuracy(
            5, _grabber('output_student'))

    metrics = {
        'loss': loss,
        'top1acc_student': acc_student,
        'top5acc_student': acc5_student
    }
    return metrics


def kd_metric_formatter(metrics):
    metrics = deepcopy(metrics)
    loss = metrics.pop('loss')
    fmtstr = f'loss: {loss["total"]:.4f} ('
    for key, value in loss.items():
        if key == 'total':
            continue
        fmtstr += f'{key}: {value:.4f}, '
    fmtstr = fmtstr[:-2] + '), '
    fmtstr += 'student top1 acc: {top1acc_student:.4f}, '
    fmtstr += 'student top5 acc: {top5acc_student:.4f}, '

    return fmtstr.format(**metrics), loss, metrics


def kd_explanation_metrics(criterion, distributed=False):
    if distributed:
        loss = DistributedMultiLoss(criterion)
        acc_student = DistributedAccuracy(
            _grabber('output_student'))
        acc5_student = DistributedTopKCategoricalAccuracy(
            5, _grabber('output_student'))
        acc_masked = DistributedAccuracy(
            _grabber('output_masked'))
        acc5_masked = DistributedTopKCategoricalAccuracy(
            5, _grabber('output_masked'))
    else:
        loss = MultiLoss(criterion)
        acc_student = Accuracy(
            _grabber('output_student'))
        acc5_student = TopKCategoricalAccuracy(
            5, _grabber('output_student'))
        acc_masked = Accuracy(
            _grabber('output_masked'))
        acc5_masked = TopKCategoricalAccuracy(
            5, _grabber('output_masked'))

    metrics = {
        'loss': loss,
        'top1acc_student': acc_student,
        'top5acc_student': acc5_student,
        'top1acc_masked': acc_masked,
        'top5acc_masked': acc5_masked,
    }
    return metrics


def kd_explanation_metric_formatter(metrics):
    metrics = deepcopy(metrics)
    loss = metrics.pop('loss')
    fmtstr = f'loss: {loss["total"]:.4f} ('
    for key, value in loss.items():
        if key == 'total':
            continue
        fmtstr += f'{key}: {value:.4f}, '
    fmtstr = fmtstr[:-2] + '), '

    fmtstr += 'student top1 acc: {top1acc_student:.4f}, '
    fmtstr += 'student top5 acc: {top5acc_student:.4f}, '
    fmtstr += 'masked top1 acc: {top1acc_masked:.4f}, '
    fmtstr += 'masked top5 acc: {top5acc_masked:.4f}, '

    return fmtstr.format(**metrics), loss, metrics


# FIXME: to catalog
_METRICS = {
    'Classification': (classification_metrics, classification_metric_formatter),
    'KDNet': (kd_metrics, kd_metric_formatter),
    'KDExplanationNet': (kd_explanation_metrics, kd_explanation_metric_formatter),
}


def create_metrics(name, criterion, distributed=False):
    metrics_getter, formatter = _METRICS[name]
    return metrics_getter(criterion, distributed), formatter
