"""

"""
import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms

from configuration.constants import AccuracyStats, LossStats, ImgSampleStats
from models.jointnet.vqvae import VQVAE
from trainer.basic_trainer import BasicTrainer
from utils.monitors.callbacks.img_sample_callback import ImgSampleCallBack
from utils.monitors.cmdline_monitor import CmdLineMonitor
from utils.monitors.tensorboard_monitor import TensorboardMonitor
from utils.monitors.callbacks.accuracy_callback import AccuracyCallback
from utils.monitors.callbacks.loss_callback import LossCallback

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
            transforms.Resize(256)
            , transforms.ToTensor()
            , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    class CIFAR10_Img_Only(torchvision.datasets.CIFAR10):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __getitem__(self, index):
            img, target = super().__getitem__(index)
            return img, img

    trainset = CIFAR10_Img_Only(
        root='./data'
        , train=True
        , download=True
        , transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data'
        , train=False
        , download=True
        , transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vqvae = VQVAE().to(device)

    train_monitors = dict()
    train_monitors[CmdLineMonitor.__name__] = CmdLineMonitor()
    train_monitors[CmdLineMonitor.__name__].add_callbacks(LossCallback())
    train_monitors[TensorboardMonitor.__name__] = TensorboardMonitor(type(vqvae).__name__)
    train_monitors[TensorboardMonitor.__name__].add_callbacks(LossCallback())
    train_monitors[TensorboardMonitor.__name__].add_callbacks(ImgSampleCallBack(len(trainloader)))


    class VQVAETrainer(BasicTrainer):
        def __init__(self):
            super().__init__()

        def generate_stats(self, inputs, targets, outputs, loss):
            generated, _ = outputs
            stats_dict = {LossStats.LOSS: loss
                , ImgSampleStats.IMG_SAMPLES: {"targets": targets, "generated": generated}}
            return stats_dict


    trainer = VQVAETrainer()
    trainer.train(vqvae
                  , trainloader
                  , vqvae.calculate_loss
                  , optim.SGD(vqvae.parameters(), lr=0.01)
                  , 2
                  , device
                  , train_monitors)


if __name__ == "__main__":
    perform_experiment()
