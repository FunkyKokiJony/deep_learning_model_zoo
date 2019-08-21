"""

"""
import pytest
import torch
from mock import Mock
from torch.utils.tensorboard import SummaryWriter

from configuration.constants import AccuracyStats, LossStats, MonitorMode
from utils.monitors.callbacks.accuracy_callback import AccuracyCallback
from utils.monitors.callbacks.loss_callback import LossCallback
from utils.monitors.tensorboard_monitor import TensorboardMonitor


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

def test_tensorboard_monitor_with_callbacks(monkeypatch, training_stats1, training_stats2):
    monitor = TensorboardMonitor("mock")
    accuracy = AccuracyCallback()
    loss = LossCallback()

    monitor.add_callbacks(accuracy)
    monitor.add_callbacks(loss)

    mock_writer = Mock()
    mock_writer.scalar_name = []
    mock_writer.scalar_val = []
    mock_writer.scalar_idx = []

    def mock_init(self, dir):
        mock_writer.dir = dir

    def mock_add_scalar(name, val, idx):
        mock_writer.scalar_name.append(name)
        mock_writer.scalar_val.append(val)
        mock_writer.scalar_idx.append(idx)

    monkeypatch.setattr(SummaryWriter, "__init__", mock_init)
    monkeypatch.setattr(mock_writer, "add_scalar", mock_add_scalar)

    with monitor:
        monkeypatch.setattr(monitor, "writer", mock_writer)
        monitor.update(MonitorMode.TRACK, 0, training_stats1)
        monitor.update(MonitorMode.EVAL, 1, training_stats2)

        assert monitor.tracking_stats[accuracy.get_name()][MonitorMode.TRACK]["idx"] == 1
        assert monitor.tracking_stats[accuracy.get_name()][MonitorMode.TRACK]["val"] == "{:.4f}".format(0.75)
        assert monitor.tracking_stats[accuracy.get_name()][MonitorMode.EVAL]["idx"] == 1
        assert monitor.tracking_stats[accuracy.get_name()][MonitorMode.EVAL]["val"] == "{:.4f}".format(1.0)

        assert monitor.tracking_stats[loss.get_name()][MonitorMode.TRACK]["idx"] == 1
        assert monitor.tracking_stats[loss.get_name()][MonitorMode.TRACK]["val"] == "{:.4f}".format(0.5)
        assert monitor.tracking_stats[loss.get_name()][MonitorMode.EVAL]["idx"] == 1
        assert monitor.tracking_stats[loss.get_name()][MonitorMode.EVAL]["val"] == "{:.4f}".format(0.0)

        monitor.display()

        assert mock_writer.scalar_name == ['_'.join(["mock", accuracy.get_name(), MonitorMode.TRACK]),
                                           '_'.join(["mock", accuracy.get_name(), MonitorMode.EVAL]),
                                           '_'.join(["mock", loss.get_name(), MonitorMode.TRACK]),
                                           '_'.join(["mock", loss.get_name(), MonitorMode.EVAL])]
        assert mock_writer.scalar_val == [0.75, 1.0, 0.5, 0.0]
        assert mock_writer.scalar_idx == [1, 1, 1, 1]
