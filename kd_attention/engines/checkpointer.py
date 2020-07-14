import os
import warnings
import logging

import torch
from ignite.handlers import ModelCheckpoint


class CheckPointer(ModelCheckpoint):
    def __init__(self, model, dirname, exper_name, save_interval, n_saved):
        super(CheckPointer, self).__init__(
            dirname,
            exper_name,
            save_interval=save_interval,
            score_function=None,
            score_name=None,
            n_saved=n_saved,
            atomic=True,
            require_empty=False,
            create_dir=True,
            save_as_state_dict=False
        )
        self._model = model
        self.exper_name = exper_name

    def forge_save(self, engine):
        return dict(
            states=dict(
                epoch=engine.state.epoch,
                model=self._model.state_dict(),
            ),
        )

    @property
    def _last_checkpoint(self):
        return os.path.join(
            get_project_root(), self._dirname, 'last_checkpoint'
        )

    def carve_trace(self):
        if self._iteration % self._save_interval == 0:
            with open(self._last_checkpoint, 'w') as f:
                f.write(
                    '{}_states_{}.pth\n'.format(
                        self._fname_prefix,
                        self._iteration
                    )
                )

    def read_trace(self):
        with open(self._last_checkpoint, 'r') as f:
            fname = f.read().replace('\n', '')
        return fname

    def __call__(self, engine):
        to_save = self.forge_save(engine)
        super(CheckPointer, self).__call__(engine, to_save)
        self.carve_trace()

    def manual_save(self, engine, suffix=''):
        to_save = self.forge_save(engine)

        for name, obj in to_save.items():
            fname = '{}_{}_{}.pth'.format(self._fname_prefix, name, suffix)
            path = os.path.join(self._dirname, fname)
            self._save(obj=obj, path=path)

    def load(self, final=False):
        if final:
            fname = '{}_states_final.pth'.format(self._fname_prefix)
        else:
            try:
                fname = self.read_trace()
            except FileNotFoundError as error:
                warnings.warn('There are no checkpoints '
                              'or checkpoint trace file.: '+str(error))
                return 0

        fname = os.path.join(get_project_root(), self._dirname, fname)
        states = torch.load(fname, map_location=torch.device('cpu'))
        new_state_dict = dict()
        parameter_count = 0
        for k in self._model.state_dict().keys():
            if k in states['model']:
                new_state_dict[k] = states['model'][k]
                parameter_count += 1
            else:
                new_state_dict[k] = self._model.state_dict()[k]
        logger = logging.getLogger(self.exper_name+'.checkpointer')
        fmtxt = 'Successfully loaded %d states from %s '
        fmtxt += '(model has %d states, loaded file has %d states)'
        logger.info(
            fmtxt,
            parameter_count,
            fname,
            len(self._model.state_dict()),
            len(states['model'])
        )
        self._model.load_state_dict(new_state_dict)
        return states['epoch']
