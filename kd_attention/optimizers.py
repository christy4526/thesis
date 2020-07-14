from torch import optim


def create_sgd_optimizer(model, lr, momentum, nesterov, weight_decay):
    optimizer = optim.SGD(
        model.parameters(),
        cfg.OPTIMIZER.LR,
        momentum=cfg.OPTIMIZER.SGD.MOMENTUM,
        nesterov=cfg.OPTIMIZER.SGD.NESTEROV,
        weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY
    )
    return optimizer


def create_adam_optimizer(model, lr, ):
    optimizer = optim.Adam(
        model.parameters(),
        cfg.OPTIMIZER.LR,
        weight_decay=cfg.OPTMIZER.WEIGHT_DECAY
    )
    return optimizer
