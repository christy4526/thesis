'''
'''
import os
import logging
import datetime
import numbers
import warnings
from collections import defaultdict

import torch

from ignite.engine import Engine
from ignite.engine import Events as E
from ignite.handlers import Timer
from ignite.handlers import ModelCheckpoint

from ..utils import get_rank
from ..utils import synchronize
from ..utils.logger import DummyLogger
from ..utils.logger import MetricLogger


class DummyCheckpointer(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


class PrerequisitesNotSatisfiedError(Exception):
    pass


class IntegratedTrainingSystem(object):
    def __init__(self, model, trainer, evaluator=None, max_epochs=1,
                 log_term=50, distributed=False, sampler=None, savedir=''):
        if not distributed ^ (sampler is None):
            raise AttributeError(
                'if you using distributed training, please pass '
                'the object of "DistributedSampler" to the __init__ argument.')
        if not isinstance(trainer, Engine):
            raise TypeError('use "Engine" class of ignite.engine instead of '
                            f'{type(trainer.__class__.__name__)}')
        if evaluator is not None and not isinstance(evaluator, Engine):
            raise TypeError('use "Engine" class of ignite.engine instead of '
                            f'{type(evaluator.__class__.__name__)}')
        self.distributed = distributed
        self.sampler = sampler
        self.model = model
        self.trainer = trainer
        self.evaluator = evaluator
        self.optimizer = None
        self.scheduler = None
        self.trainloader = None
        self.evalloader = None
        self.checkpointer = None
        self.max_epochs = max_epochs
        self.log_term = log_term
        self.metric_logger = MetricLogger()
        self._decorated = False
        self._fmtr = str
        self.to_save = None
        self.savedir = savedir

    def is_mainprocess(self):
        return get_rank() == 0

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def set_dataloaders(self, trainloader, evalloader):
        self.trainloader = trainloader
        self.evalloader = evalloader

    def set_evaluation_formatter(self, formatter):
        self._fmtr = formatter

    def set_checkpointer(self, filename_prefix, to_save=None,
                         save_interval=None, create_dir=True):
        self.checkpointer = None
        self.to_save = to_save or self.model

        def save_checkpoint(engine):
            if engine.state.epoch % save_interval == 0:
                states = {'model': self.to_save.state_dict(),
                          'optimizer': self.optimizer.state_dict()}
                if self.scheduler is not None:
                    scheduler = self.scheduler
                    while not hasattr(scheduler, 'state_dict'):
                        scheduler = scheduler.lr_scheduler
                    states['scheduler'] = scheduler.state_dict()
                filepath = os.path.join(
                    self.savedir,
                    f'{filename_prefix}_{engine.state.epoch}.pth'
                )
                torch.save(states, filepath)
        self.checkpointer = save_checkpoint

    def save_state(self, workspace_path, prefix=''):
        raise NotImplementedError

    def load_state(self, workspace_path, prefix=''):
        raise NotImplementedError

    def _get_logger(self, name):
        if self.is_mainprocess():
            logger = logging.getLogger(name)
        else:
            logger = DummyLogger()
        return logger

    def _update_metric(self, output):
        if isinstance(output, torch.Tensor):
            warnings.warn(f'stacking {torch.Tensor.__name__} may cause'
                          'unnecessary memory occupation')
            self.metric_logger.update(output=output)
        elif isinstance(output, dict):
            self.metric_logger.update(**output)
        elif isinstance(output, (list, tuple)):
            kwargs = dict()
            for i, o in enumerate(output):
                kwargs[f'loss{i}'] = o
            self.metric_logger.update(**kwargs)
        elif isinstance(output, numbers.Number):
            self.metric_logger.update(output=output)
        else:
            raise TypeError(f'unsupported type {type(output).__name__}')

    def _decorate_defaults(self):
        timer = Timer(average=True)
        timer.attach(self.trainer, step=E.ITERATION_COMPLETED)

        @self.trainer.on(E.EPOCH_STARTED)
        def start_epoch(engine):
            epoch = engine.state.epoch
            logger = self._get_logger('training')
            logger.info('epoch %d training started', epoch)
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)

        if self.scheduler is not None:
            self.trainer.add_event_handler(
                E.ITERATION_COMPLETED, self.scheduler
            )

        if self.distributed:
            @self.trainer.on(E.ITERATION_STARTED)
            def synchronize_models(engine):
                synchronize()

        @self.trainer.on(E.ITERATION_COMPLETED)
        def log_results(engine):
            self._update_metric(engine.state.output)

            global_iter = engine.state.iteration
            if global_iter % self.log_term == 0:
                logger = logging.getLogger('training')
                epoch = engine.state.epoch
                local_iter = global_iter - (epoch - 1) * len(self.trainloader)
                iter_remains = self.max_epochs * \
                    len(self.trainloader) - global_iter

                elapse = timer._elapsed()
                elapse = str(datetime.timedelta(seconds=int(elapse)))

                eta = iter_remains * timer.value()
                eta = str(datetime.timedelta(seconds=int(eta)))

                logstr = f'elapse: {elapse} eta: {eta} '
                logstr += f'Epoch [{epoch}/{self.max_epochs}] '
                logstr += f'[{local_iter}/{len(self.trainloader)}] ({global_iter}) '
                logstr += f'lr: {self.optimizer.param_groups[0]["lr"]:.3e} '
                logstr += str(self.metric_logger)
                logger.info(logstr)

        if self.evaluator is not None and self.evalloader is not None:
            @self.trainer.on(E.EPOCH_COMPLETED)
            def evaluate(engine):
                epoch = engine.state.epoch
                logger = self._get_logger('evaluation')
                logger.info('epoch %d evaluation started', epoch)
                self.evaluator.run(self.evalloader)
                result = self._fmtr(self.evaluator.state.metrics)
                logger.info('epoch %d evaluation results: \n%s', epoch, result)

        if self.checkpointer is not None:
            self.trainer.add_event_handler(
                E.EPOCH_COMPLETED, self.checkpointer
            )

        if self.is_mainprocess():
            @self.trainer.on(E.COMPLETED)
            def save_final(engine):
                model = self.model
                if hasattr(model, 'module'):
                    model = model.module
                torch.save(
                    model.student.cpu().state_dict(),
                    os.path.join(self.savedir, 'final_model.pth')
                )

    def decorate(self, trainer_decorators={}, evaluator_decorators={}):
        self._decorate_defaults()
        self._decorated = True

        t_decorators = defaultdict(list)
        e_decorators = defaultdict(list)
        t_decorators.update(trainer_decorators)
        e_decorators.update(evaluator_decorators)

        for event in E:
            for handler in t_decorators[event]:
                self.trainer.add_event_handler(
                    event, handler
                )
            for handler in e_decorators[event]:
                self.evaluator.add_event_handler(
                    event, handler
                )

    def run(self):
        if self.trainloader is None:
            raise PrerequisitesNotSatisfiedError(
                '.set_dataloaders is not called')

        if not self._decorated:
            if self.is_mainprocess():
                warnings.warn('decoration is omitted. '
                              'run "IntegratedTrainingSystem.decorate" to get '
                              'furthermore information of training process.')
        if self.checkpointer is None:
            if self.is_mainprocess():
                warnings.warn('if checkpointer is not created, '
                              'the training process'
                              'will be evaporated after training.')
        return self.trainer.run(self.trainloader, self.max_epochs)
