"""

"""
import logging

from configuration.settings import LossStats, MonitorMode
from utils.monitors.callbacks.stats_callback import StatsCallback


class LossCallback(StatsCallback):
    def __init__(self):
        self.tracking_loss = 0
        self.total = 0

    def __call__(self, monitor, mode, idx, stats_dict):
        _loss = stats_dict.get(LossStats.LOSS).item()
        if _loss is None:
            logging.error("Key [{}] is not set".format(LossStats.LOSS))

        if mode == MonitorMode.TRACK:
            self.tracking_loss += _loss
            self.total += 1
        elif mode == MonitorMode.EVAL:
            monitor.add_stats(idx, self.get_name(), MonitorMode.TRACK, "{:.4f}".format(self.tracking_loss / self.total))
            monitor.add_stats(idx, self.get_name(), MonitorMode.EVAL, "{:.4f}".format(_loss))
            self.tracking_loss = 0
            self.total = 0
        else:
            logging.error("Unknown mode for callback!")

    def get_name(self):
        return "loss"