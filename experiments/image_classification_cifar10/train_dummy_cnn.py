"""
This module is for trainer and evaluation of DummyCNN on cifar10 dataset
"""
import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
from models.dummynet import dummy_cnn
from trainer.basic_trainer import BasicTrainer
from utils.monitors.cmdline_monitor import CmdLineMonitor
from utils.monitors.tensorboard_monitor import TensorboardMonitor
from utils.monitors.callbacks.accuracy_callback import AccuracyCallback
from utils.monitors.callbacks.loss_callback import LossCallback
from trainer.deprecation.basic_training import BasicTraining

def perform_experiment():
    """
    1. Load cifar10 dataset into trainer dataloader and test dataloader
    2. Instantiate ResNet model and BasicTraining class
    3. Start trainer and evaluation
    """
    # transform PIL image from [0, 1] to [-1, 1]
    # the first tuple is the means for 3 different channels
    # the second tuple is standard deviations. (pixel value - mean) / std val
    transform = transforms.Compose(
        [
            transforms.Resize(227)
            , transforms.ToTensor()
            , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root='./data'
        , train=True
        , download=True
        , transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data'
        , train=False
        , download=True
        , transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = dummy_cnn.DummyCNN().to(device)

    train_monitors = dict()
    train_monitors[CmdLineMonitor.__name__] = CmdLineMonitor()
    train_monitors[CmdLineMonitor.__name__].add_callbacks(AccuracyCallback())
    train_monitors[CmdLineMonitor.__name__].add_callbacks(LossCallback())
    train_monitors[TensorboardMonitor.__name__] = TensorboardMonitor(type(cnn).__name__)
    train_monitors[TensorboardMonitor.__name__].add_callbacks(AccuracyCallback())
    train_monitors[TensorboardMonitor.__name__].add_callbacks(LossCallback())

    test_monitors = dict()
    test_monitors[CmdLineMonitor.__name__] = CmdLineMonitor()
    test_monitors[CmdLineMonitor.__name__].add_callbacks(AccuracyCallback())
    #It is better to have a different name for TensorboardMonitor because it determine the runs folder
    # and one run supposed only have one writer (one monitor)
    test_monitors[TensorboardMonitor.__name__] = TensorboardMonitor(type(cnn).__name__ + "_test")
    test_monitors[TensorboardMonitor.__name__].add_callbacks(AccuracyCallback())

    trainer = BasicTrainer()
    trainer.train(cnn
                  , trainloader
                  , nn.CrossEntropyLoss()
                  , optim.SGD(cnn.parameters(), lr=0.01)
                  , 2
                  , device
                  , train_monitors
                  , checkpoint_name="checkpoint"
                  , checkpoint_interval=2 * len(trainloader))

    trainer.eval(cnn
                 , testloader
                 , device
                 , test_monitors)

    test_monitors[TensorboardMonitor.__name__].flush()

    # training = BasicTraining()
    # training.train(cnn
    #                , trainloader
    #                , nn.CrossEntropyLoss()
    #                , optim.SGD(cnn.parameters(), lr=0.01)
    #                , 4
    #                , device
    #                , loss_display_interval=2000
    #                , tensorboardx_loss_display_interval=10
    #                , accuracy_display_batch=2000)
    # training.eval(cnn, testloader, device)

if __name__ == "__main__":
    perform_experiment()
