'''
DEPRICATED: may not be needed
'''
import torch

from ignite.engine import Engine
from ignite.engine import _prepare_batch


def _output_transform(x, y, y_preds):
    return y_preds, y


def create_multi_loss_supervised_evaluator(
        model, metrics=None, device=None, non_blocking=False,
        prepare_batch=_prepare_batch, output_transform=_output_transform):
    metrics = metrics or {}

    if device:
        model.to(device)

    @torch.no_grad()
    def _inference(engine, batch):
        model.eval()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_preds = model(x)
        return output_transform(x, y, y_preds)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
