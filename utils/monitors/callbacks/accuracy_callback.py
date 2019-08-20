"""

"""
from configuration.settings import AccuracyStats, MonitorMode
from utils.monitors.callbacks.stats_callback import StatsCallback
import logging

class AccuracyCallback(StatsCallback):

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, monitor, mode, idx, stats_dict):
        _labels = stats_dict.get(AccuracyStats.LABELS)
        _predicts = stats_dict.get(AccuracyStats.PREDICTS)

        if _labels is None or _predicts is None:
            logging.error("Key [{labels}] or [{predicts}] is not set"
                          .format(labels=AccuracyStats.LABELS)
                            , predicts=AccuracyStats.PREDICTS)
            return

        if mode == MonitorMode.TRACK:
            self.correct += (_predicts == _labels).sum().item()
            self.total += _labels.size(0)
        elif mode == MonitorMode.EVAL:
            _correct = (_predicts == _labels).sum().item()
            _total = _labels.size(0)
            monitor.add_stats(idx, self.get_name(), MonitorMode.TRACK, "{:.4f}%".format(self.correct / self.total))
            monitor.add_stats(idx, self.get_name(), MonitorMode.EVAL, "{:.4f}%".format(_correct / _total))
            self.correct = 0
            self.total = 0
        else:
            logging.error("Unknown mode for callback!")

    def get_name(self):
        return "accuracy"