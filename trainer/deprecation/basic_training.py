"""
This module is for the trainer and evaluation of basic image classification with tracking loss
"""
import torch
from utils.deprecation.loss_monitor import LossMonitor
from utils.deprecation.accuracy_monitor import AccuracyMonitor

class BasicTraining:
    """
    This class is for the trainer and evaluation of image classification procedure
    """
    def __init__(self):
        self.loss_monitor = LossMonitor()
        self.accuracy_monitor = AccuracyMonitor()

    def train(self, model, trainloader, criterion, optimizer, epochs, device
              , loss_display_interval=1000
              , tensorboardx_loss_display_interval=10
              , accuracy_display_batch=5000):
        """
        1. This method is for trainer procedure
        2. Contain trainer loss and accuracy tracking on command line and tensorboard
        :param model: pytorch model
        :param trainloader: dataloader for trainer set
        :param criterion: method for calculate loss
        :param optimizer: method for parameter optimizer
        :param epochs: how many epochs we would like to train
        :param device: the trainer device, either cpu or gpu
        :param loss_display_interval: the interval for display loss on command line
        :param tensorboardx_loss_display_interval: the interval for display loss on tensorboard
        :param accuracy_display_batch: the interval for display validation accuracy on tensorboard
        """
        self.loss_monitor.commandline_loss_init()
        self.loss_monitor.tensorboardx_loss_init(type(model).__name__)
        self.accuracy_monitor.commandline_accuracy_init()
        self.accuracy_monitor.tensorboardx_accuracy_init(type(model).__name__)

        for epoch in range(epochs):
            for idx, data in enumerate(trainloader):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                model.train()
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                self.loss_monitor.commandline_loss_monitor(
                    epoch, idx, loss.item(), trainloader, loss_display_interval)
                self.loss_monitor.tensorboardx_loss_monitor(
                    epoch, idx, loss.item(), trainloader, tensorboardx_loss_display_interval)
                #Since .data is for Variable to underlying tensor which has been deprecated
                #max(outputs, 1) will return the max value and their indices
                _, predicted = torch.max(outputs, 1)
                self.accuracy_monitor.commandline_accuracy_monitor(
                    labels, predicted
                    , tracking_mode=True
                    , epoch=epoch
                    , idx=idx
                    , dataloader=trainloader
                    , interval=accuracy_display_batch)
                self.accuracy_monitor.tensorboardx_accuracy_monitor(
                    labels, predicted
                    , tracking_mode=True
                    , epoch=epoch
                    , idx=idx
                    , dataloader=trainloader
                    , interval=accuracy_display_batch)


    def eval(self, model, testloader, device):
        """
        1. This is the method for evaluation on test set
        2. Contain outputting final model test accuracy on command line and tensorboard
        :param model: pytorch model
        :param testloader: the dataloader for test set
        :param device: the device we run our model, either cpu or gpu
        """
        self.accuracy_monitor.commandline_accuracy_init()
        self.accuracy_monitor.tensorboardx_accuracy_init(type(model).__name__)

        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                model.eval()
                outputs = model.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                self.accuracy_monitor.commandline_accuracy_monitor(labels, predicted)
                self.accuracy_monitor.tensorboardx_accuracy_monitor(labels, predicted)

        self.accuracy_monitor.commandline_accuracy_monitor(display=True)
        self.accuracy_monitor.tensorboardx_accuracy_monitor(display=True)
