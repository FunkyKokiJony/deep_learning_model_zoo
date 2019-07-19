"""
This module is for using command ling and tensorboardx to track loss
"""
import time
from configuration import settings
from tensorboardX import SummaryWriter

class LossMonitor:
    """
    This class is for using command ling and tensorboardx to track loss
    """
    __LOG_ADDR = settings.RESOURCES_ADDR + '/log/'

    def __init__(self):
        self.runningloss_commandline = 0
        self.total_commandline = 0
        self.start_time = None
        self.writer = None
        self.runningloss_tensorboard = 0
        self.total_tensorboard = 0
        self.tensorboardx_name = None

    def commandline_loss_init(self):
        """
        Reset the variables for display loss on command line
        """
        self.runningloss_commandline = 0
        self.total_commandline = 0

    def tensorboardx_loss_init(self, tensorboardx_name="Undefined"):
        """
        Reset the variables for output loss to tensorboard
        This method will record the timestamp of init and used as an id for tensorboard runs
        :param tensorboardx_name: the tag name used for tensorboard
        """
        self.start_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
        self.writer = SummaryWriter(self.__LOG_ADDR + self.start_time + "_loss")

        self.runningloss_tensorboard = 0
        self.total_tensorboard = 0
        self.tensorboardx_name = tensorboardx_name
        print('loss monitor run timestamp: ' + self.start_time)

    def commandline_loss_monitor(self, epoch, idx, loss, dataloader, interval=2000, display=False):
        """
        1. Method for display loss on the command line
        2. It will output average loss every [interval] times
        :param epoch: current epoch in training
        :param idx: current idx in current epoch
        :param loss: the loss numeric value in this training batch
        :param dataloader: the dataloader we used in training
        :param interval: loss display interval
        :param display: boolean determine if display loss using accumulate statistics
        """
        self.runningloss_commandline += loss
        self.total_commandline += 1
        if idx % interval == interval - 1 or display:
            niter = epoch * len(dataloader) + idx
            print("[{:d}, {:d}] loss: {:.3f}".format(
                epoch, idx, self.runningloss_commandline / self.total_commandline))
            self.runningloss_commandline = 0
            self.total_commandline = 0

    def tensorboardx_loss_monitor(self, epoch, idx, loss, dataloader, interval=10, display=False):
        """
        1. Method for display loss on the tensorboard with a scalar graph
        2. It will output average loss every [interval] times, this interval should be small
        3. The tag of scalar graph is defined by [tensorboardx_name]
        4. The name of runs is defined by the timestamp assigned with corresponding init method
        :param epoch:
        :param idx:
        :param loss:
        :param dataloader:
        :param interval:
        :param display:
        :return:
        """
        self.runningloss_tensorboard += loss
        self.total_tensorboard += 1
        if idx % interval == interval - 1 or display:
            niter = epoch * len(dataloader) + idx
            self.writer.add_scalar(self.tensorboardx_name + "_Loss"
                                   , self.runningloss_tensorboard / self.total_tensorboard, niter)
            self.runningloss_tensorboard = 0
            self.total_tensorboard = 0
            self.writer.flush()
