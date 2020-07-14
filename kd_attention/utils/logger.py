import logging
import os

from collections import deque
from collections import defaultdict

import numpy as np


class FakePbar:
    def update(self, *args, **kwargs):
        return

    def close(self, *args, **kwargs):
        return

    def set_description(self, *args, **kwargs):
        return

    def write(self, *args, **kwargs):
        return


class DummyLogger(object):
    def debug(self, *args, **kwargs):
        return

    def info(self, *args, **kwargs):
        return

    def warning(self, *args, **kwargs):
        return

    def error(self, *args, **kwargs):
        return

    def critical(self, *args, **kwargs):
        return


class SmoothedValue(object):
    def __init__(self, window_size=50):
        self.deque = deque(maxlen=window_size)
        self.count = 0

    def update(self, value):
        self.deque.append(value)

    @property
    def median(self):
        if len(self.deque) == 0:
            return 0
        return np.median(self.deque)

    @property
    def mean(self):
        if len(self.deque) == 0:
            return 0
        return np.mean(self.deque)


class MetricLogger(object):
    def __init__(self, delimiter=' ', fmt=':.4f'):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.fmt = fmt

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(v)

    def __getattr__(self, name):
        if name in self.meters:
            return self.meters[name]
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")

    def __str__(self):
        values = []
        for name, meter in self.meters.items():
            fmtstr = '{name}: {val'+self.fmt+'}'
            values.append(fmtstr.format(name=name, val=meter.median))
        return self.delimiter.join(values)


def setup_logger(name, save_dir, rank, filename='log.txt'):
    if rank != 0:
        return DummyLogger()
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt='%(asctime)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(save_dir, filename))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)
    return _logger


if __name__ == '__main__':
    logger = setup_logger('me', './', 0, 'debuglog.log')
    logger.debug('hello')
