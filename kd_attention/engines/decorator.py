import datetime
from ignite.engine import Events as E
from ignite.handlers import Timer
from ..utils import get_rank


def decorate_trainer(trainer, evaluator, scheduler, trainloader, valloader,
                     sampler, workspace_dir, logfilename, logterm,
                     max_epochs, checkpointer, model):
    trainer.setup_logger('trainer', workspace_dir, get_rank(), logfilename)
    trainer.setup_metric_logger()
    evaluator.setup_logger('evaluator', workspace_dir, get_rank(), logfilename)

    timer = Timer(average=True)
    timer.attach(trainer, step=E.ITERATION_COMPLETED)

    @trainer.on(E.EPOCH_STARTED)
    def start_epoch(engine):
        epoch = engine.state.epoch
        engine.logger.info(f'start training epoch {epoch}')
        if sampler is not None:
            sampler.set_epoch(epoch)

    trainer.add_event_handler(E.ITERATION_COMPLETED, scheduler)

    @trainer.on(E.ITERATION_COMPLETED)
    def log_results(engine):
        engine.metric_logger.update(**engine.state.output)
        global_iter = engine.state.iteration
        if global_iter % logterm == 0:
            epoch = engine.state.epoch
            iter_per_epoch = len(trainloader)
            local_iter = global_iter - (engine.state.epoch-1) * iter_per_epoch
            iter_remain = max_epochs * iter_per_epoch - global_iter

            elapse = timer._elapsed()
            elapse = str(datetime.timedelta(seconds=int(elapse)))

            eta = iter_remain * timer.value()
            eta = str(datetime.timedelta(seconds=int(eta)))

            logstr = f'elapse: {elapse} eta: {eta} '
            logstr += f'Epoch [{epoch}/{max_epochs}] '
            logstr += f'[{local_iter}/{iter_per_epoch}] ({global_iter}) '
            logstr += f'lr: {engine.optimizer.param_groups[0]["lr"]:.3e} '
            logstr += str(engine.metric_logger)
            engine.logger.info(logstr)

    if evaluator is not None and valloader is not None:
        @trainer.on(E.EPOCH_COMPLETED)
        def validate(engine):
            epoch = engine.state.epoch
            evaluator.logger.info(f'start evaluation epoch {epoch}')
            evaluator.run()

            result = evaluator.state.metrics
            fmtstr = f'Epoch {epoch} validation result: '
            for k, v in result.items():
                fmtstr += f'{k}: {v:.4f} '
            evaluator.logger.info(fmtstr)

    if get_rank() == 0:
        trainer.add_event_handler(
            E.EPOCH_COMPLETED, checkpointer, {'epoch', model}
        )
