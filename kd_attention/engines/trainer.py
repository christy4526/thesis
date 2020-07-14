'''
to torchtool?
'''

from ignite.engine import Engine
from ignite.engine import _prepare_batch


def _reduce_loss_dict(loss_dict):
    total = 0
    for k, v in loss_dict.items():
        total += v
    return total


def _output_transform(x, y, y_preds, losses):
    loss_items = dict()
    for k, v in losses.items():
        loss_items[k] = v.item()

    return loss_items


def create_multi_loss_supervised_trainer(
        model,
        optimizer,
        loss_fn,
        device=None,
        non_blocking=False,
        prepare_batch=_prepare_batch,
        output_transform=_output_transform,
        reduce_function=_reduce_loss_dict):
    if device is not None:
        model.to(device)

    def _update(engine, batch):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_preds = model(x)
        losses = loss_fn(y_preds, y)

        loss = reduce_function(losses)
        loss.backward()
        optimizer.step()

        if isinstance(losses, dict):
            losses['total'] = loss
        elif isinstance(losses, (list, tuple)):
            losses.append(loss)
        return output_transform(x, y, y_preds, losses)

    return Engine(_update)
