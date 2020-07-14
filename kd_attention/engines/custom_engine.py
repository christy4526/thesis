from ignite.engine import Engine as _Engine
from ..utils import setup_logger
from ..utils.logger import DummyLogger
from ..utils.logger import MetricLogger


class Engine(_Engine):
    def __init__(self, process_function, dataloader,
                 max_epochs, optimizer=None):
        super().__init__(process_function)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.logger = DummyLogger()
        self.metric_logger = None

    def setup_logger(self, name, save_dir, rank, filename):
        self.logger = setup_logger(name, save_dir, rank, filename)

    def setup_metric_logger(self, delimiter=' ', fmt=':.4f'):
        self.metric_logger = MetricLogger(delimiter, fmt)

    def run(self):
        return super().run(self.dataloader, self.max_epochs)
