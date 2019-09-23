"""

"""
import torch
import tqdm

from configuration.constants import MonitorMode, AccuracyStats, LossStats
from utils.monitors.cmdline_monitor import CmdLineMonitor
from utils.monitors.tensorboard_monitor import TensorboardMonitor
from utils.trainer_utils.checkpoint_utils import CheckpointHandler


class BasicTrainer:
    def __init__(self):
        pass

    def generate_stats(self, inputs, targets, outputs, loss):
        _, predicted = torch.max(outputs, 1)
        stats_dict = {AccuracyStats.LABELS: targets,
                      AccuracyStats.PREDICTS: predicted,
                      LossStats.LOSS: loss}
        return stats_dict

    def train(self, model, trainloader, criterion, optimizer, epochs, device
              , monitors=dict(), checkpoint_name=None, checkpoint_interval=1000
              , tensorboard_eval_interval=10):

        checkpoint_handler = CheckpointHandler(model, optimizer)
        cmdline_eval_interval = len(trainloader)
        it = 0

        with tqdm.trange(1, epochs + 1, desc="epochs") as epoch_bar:
            for epoch in epoch_bar:
                with tqdm.tqdm(trainloader, leave=False, desc="train") as idx_bar:
                    for data in idx_bar:
                        inputs, targets = data[0].to(device), data[1].to(device)
                        optimizer.zero_grad()
                        model.train()
                        outputs = model.forward(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()

                        stats_dict = self.generate_stats(inputs, targets, outputs, loss)

                        it += 1

                        if checkpoint_name is not None and (it % checkpoint_interval) == 0 and it != 0:
                            checkpoint_handler.save_checkpoint(
                                checkpoint_name + "_" + str(it) + "_iteration"
                                , epoch, it, loss)

                        for _, monitor in monitors.items():
                            monitor.update(MonitorMode.TRACK, it, stats_dict)

                        if TensorboardMonitor.__name__ in monitors\
                                    and (it % tensorboard_eval_interval) == 0 and it != 0:
                            monitors[TensorboardMonitor.__name__].update(MonitorMode.DISPLAY
                                                                          , it, stats_dict)
                            monitors[TensorboardMonitor.__name__].display()

                        if CmdLineMonitor.__name__ in monitors \
                                and (it % cmdline_eval_interval) == 0 and it != 0:
                            monitors[CmdLineMonitor.__name__].update(MonitorMode.DISPLAY
                                                                      , it, stats_dict)

                            monitors[CmdLineMonitor.__name__].display()
                            monitors[CmdLineMonitor.__name__].reset()

        if checkpoint_name is not None:
            checkpoint_handler.save_checkpoint(checkpoint_name + "_" + str(epochs) + "_epochs")

    def eval(self, model, testloader, device, monitors):
        with torch.no_grad(), tqdm.tqdm(testloader, desc="test") as idx_bar:
            it = 0
            for data in idx_bar:
                it += 1
                inputs, labels = data[0].to(device), data[1].to(device)
                model.eval()
                outputs = model.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)

                stats_dict = {AccuracyStats.LABELS: labels,
                              AccuracyStats.PREDICTS: predicted}

                for _, monitor in monitors.items():
                    monitor.update(MonitorMode.TRACK, it, stats_dict)

            for _, monitor in monitors.items():
                monitor.update(MonitorMode.DISPLAY, it)
                monitor.display("Test Accuracy")
