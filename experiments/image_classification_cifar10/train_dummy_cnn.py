"""
This module is for training and evaluation of DummyCNN on cifar10 dataset
"""
import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
from models.dummynet import DummyCNN
from training.basic_training import BasicTraining

def perform_experiment():
    """
    1. Load cifar10 dataset into training dataloader and test dataloader
    2. Instantiate ResNet model and BasicTraining class
    3. Start training and evaluation
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
    dummy_cnn = DummyCNN.DummyCNN().to(device)
    training = BasicTraining()
    training.train(dummy_cnn
                   , trainloader
                   , nn.CrossEntropyLoss()
                   , optim.SGD(dummy_cnn.parameters(), lr=0.01)
                   , 4
                   , device
                   , loss_display_interval=2000
                   , tensorboardx_loss_display_interval=10
                   , accuracy_display_batch=2000)
    training.eval(dummy_cnn, testloader, device)

if __name__ == "__main__":
    perform_experiment()
