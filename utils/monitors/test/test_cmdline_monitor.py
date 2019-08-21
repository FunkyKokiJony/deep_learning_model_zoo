"""

"""
import pytest
import torch

from configuration.constants import AccuracyStats, LossStats, MonitorMode
from utils.monitors.callbacks.accuracy_callback import AccuracyCallback
from utils.monitors.callbacks.loss_callback import LossCallback
from utils.monitors.cmdline_monitor import CmdLineMonitor


@pytest.fixture
def training_stats1():
    stats_dict = dict()
    stats_dict[AccuracyStats.LABELS] = torch.tensor([1, 2, 3, 4, 2, 3, 4 ,5])
    stats_dict[AccuracyStats.PREDICTS] = torch.tensor([1, 2, 3, 3, 2, 3, 4, 4])
    stats_dict[LossStats.LOSS] = torch.tensor([0.5])
    return stats_dict

@pytest.fixture
def training_stats2():
    stats_dict = dict()
    stats_dict[AccuracyStats.LABELS] = torch.tensor([1, 2, 3, 4, 2, 3, 4 ,5])
    stats_dict[AccuracyStats.PREDICTS] = torch.tensor([1, 2, 3, 4, 2, 3, 4, 5])
    stats_dict[LossStats.LOSS] = torch.tensor([0.0])
    return stats_dict

def test_cmdline_motnior_with_callbacks(training_stats1, training_stats2):
    monitor = CmdLineMonitor()
    accuracy = AccuracyCallback()
    loss = LossCallback()

    monitor.add_callbacks(accuracy)
    monitor.add_callbacks(loss)
    monitor.update(MonitorMode.TRACK, 0, training_stats1)

    assert accuracy.correct == 6
    assert accuracy.total == 8
    assert loss.tracking_loss == 0.5
    assert loss.total == 1

    monitor.update(MonitorMode.EVAL, 1, training_stats2)

    assert monitor.tracking_stats[accuracy.get_name()][MonitorMode.TRACK] == "{:.4f}".format(6 / 8)
    assert monitor.tracking_stats[accuracy.get_name()][MonitorMode.EVAL] == "{:.4f}".format(1.0)
    assert monitor.tracking_stats[loss.get_name()][MonitorMode.TRACK] == "{:.4f}".format(0.5)
    assert monitor.tracking_stats[loss.get_name()][MonitorMode.EVAL] == "{:.4f}".format(0.0)

    monitor.display()
    monitor.clear()



