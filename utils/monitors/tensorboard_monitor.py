"""

"""
from torch.utils.tensorboard import SummaryWriter

from configuration.config import Config
from configuration.constants import TensorboardConfig
import time


class TensorboardMonitor:
    def __init__(self, name="Undefined"):
        self.callbacks = dict()
        self.tracking_stats = dict()
        self.board_name = name
        self.cfg = Config()
        self.logdir = self.cfg.get(TensorboardConfig.SECTION
                                   , TensorboardConfig.LOGDIR)

    def __enter__(self):
        self.writer = SummaryWriter(self.logdir + time.strftime('%Y%m%d%H%M%S', time.localtime()))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    def add_callbacks(self, cb):
        self.callbacks[cb.get_name()] = cb

    def update(self, mode, idx, stats_dict):
        for _, callback in self.callbacks.items():
            callback(self, mode, idx, stats_dict)

    def add_stats(self, idx, stats_name, tag, val):
        if stats_name not in self.tracking_stats:
            self.tracking_stats[stats_name] = dict()

        self.tracking_stats[stats_name][tag] = {"idx": idx, "val": val}

    def display(self):
        if (len(self.tracking_stats) == 0): return

        for name, stats in self.tracking_stats.items():
            if len(stats) == 0:
                continue

            for tag, pair in stats.items():
                self.writer.add_scalar('_'.join([self.board_name, name, tag])
                                       , float(pair["val"]), pair["idx"])

    def clear(self):
        self.tracking_stats.clear()
        self.writer.close()
        self.writer = SummaryWriter(self.logdir + time.strftime('%Y%m%d%H%M%S', time.localtime()))




