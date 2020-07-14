from ignite.contrib.handlers import LRScheduler
from ignite.contrib.handlers import param_scheduler

from torch.optim.lr_scheduler import MultiStepLR


def build_lr_scheduler(cfg, optimizer, ngpus, iterations_per_epoch):
    scheduler = LRScheduler(MultiStepLR(
        optimizer,
        [milestone * iterations_per_epoch
         for milestone in cfg.SCHEDULER.MILESTONES],
        gamma=cfg.SCHEDULER.GAMMA,
    ))

    needs_warming_up = ngpus > 8
    if needs_warming_up:
        start_lr = optimizer.defaults['lr']
        lr = optimizer.defaults['lr'] * ngpus * 0.125
        scheduler = param_scheduler.create_lr_scheduler_with_warmup(
            scheduler,
            warmup_start_value=start_lr,
            warmup_end_value=lr,
            warmup_duration=iterations_per_epoch * 5
        )

    return scheduler
