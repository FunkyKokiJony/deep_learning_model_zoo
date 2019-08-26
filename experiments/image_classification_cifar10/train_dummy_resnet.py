"""
This module is for trainer and evaluation of DummyCNN on cifar10 dataset
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models.dummynet import dummy_resnet
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
            transforms.Resize(224)
            , transforms.ToTensor()
            , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root='./data'
        , train=True
        , download=True
        , transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data'
        , train=False
        , download=True
        , transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet = dummy_resnet.DummyResNet().to(device)
    training = BasicTraining()
    training.train(resnet
                   , trainloader
                   , nn.CrossEntropyLoss()
                   , optim.SGD(resnet.parameters(), lr=0.01)
                   , 32
                   , device
                   , loss_display_interval=10000 // trainloader.batch_size
                   , tensorboardx_loss_display_interval=10
                   , accuracy_display_batch=10000 // trainloader.batch_size)
    training.eval(resnet, testloader, device)

if __name__ == "__main__":
    perform_experiment()
