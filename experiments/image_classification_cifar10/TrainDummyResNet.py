import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from models.dummynet import DummyResNet
from training.BasicTraining import BasicTraining

if __name__ == "__main__":
    # transform PIL image from [0, 1] to [-1, 1], the first tuple is the means for 3 different channels
    # the second tuple is standard deviations. (pixel value - mean) / std val
    transform = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dummyResNet = DummyResNet.DummyResNet().to(device)
    training = BasicTraining()
    training.train(dummyResNet
                   , trainloader
                   , nn.CrossEntropyLoss()
                   , optim.SGD(dummyResNet.parameters(), lr=0.01)
                   , 32
                   , device
                   , lossDisplayBatch=10000 // trainloader.batch_size
                   , tensorboardxLossDisplayBatch=10
                   , accuracyDisplayBatch=10000 // trainloader.batch_size)
    training.eval(dummyResNet, testloader, device)
