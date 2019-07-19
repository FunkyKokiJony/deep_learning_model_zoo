from utils.loss_monitor import LossMonitor
from utils.accuracy_monitor import AccuracyMonitor
import torch

class BasicTraining:
    def __init__(self):
        self.lossMonitor = LossMonitor()
        self.accuracyMonitor = AccuracyMonitor()

    def train(self, model, trainloader, criterion, optimizer, epochs, device, lossDisplayBatch=1000, tensorboardxLossDisplayBatch=10, accuracyDisplayBatch=5000):
        self.lossMonitor.commandline_loss_init()
        self.lossMonitor.tensorboardx_loss_init(type(model).__name__)
        self.accuracyMonitor.commandline_accuracy_init()
        self.accuracyMonitor.tensorboardx_accuracy_init(type(model).__name__)

        for epoch in range(epochs):
            for idx, data in enumerate(trainloader):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                model.train()
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                self.lossMonitor.commandline_loss_monitor(epoch, idx, loss.item(), trainloader, lossDisplayBatch)
                self.lossMonitor.tensorboardx_loss_monitor(epoch, idx, loss.item(), trainloader, tensorboardxLossDisplayBatch)
                #Since .data is for Variable to underlying tensor which has been deprecated
                #max(outputs, 1) will return the max value at first and their indices at the second variable
                _, predicted = torch.max(outputs, 1)
                self.accuracyMonitor.commandline_accuracy_monitor(labels, predicted
                                                                  , tracking_mode=True, epoch=epoch
                                                                  , idx=idx, dataloader=trainloader
                                                                  , interval=accuracyDisplayBatch)
                self.accuracyMonitor.tensorboardx_accuracy_monitor(labels, predicted
                                                                   , tracking_mode=True, epoch=epoch
                                                                   , idx=idx, dataloader=trainloader
                                                                   , interval=accuracyDisplayBatch)


    def eval(self, model, testloader, device):
        self.accuracyMonitor.commandline_accuracy_init()
        self.accuracyMonitor.tensorboardx_accuracy_init(type(model).__name__)

        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                model.eval()
                outputs = model.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                self.accuracyMonitor.commandline_accuracy_monitor(labels, predicted)
                self.accuracyMonitor.tensorboardx_accuracy_monitor(labels, predicted)

        self.accuracyMonitor.commandline_accuracy_monitor(display=True)
        self.accuracyMonitor.tensorboardx_accuracy_monitor(display=True)
