"""
This module is for using command ling and tensorboardx to track accuracy
"""
import time
import torch
from configuration import settings
from tensorboardX import SummaryWriter

class AccuracyMonitor:
    """
    This class is for use command ling and tensorboardx to track accuracy
    """
    __LOG_ADDR = settings.RESOURCES_ADDR + '/log/'

    def __init__(self):
        self.correct_commandline = 0
        self.total_commandline = 0
        self.start_time = None
        self.writer = None
        self.correct_tensorboard = 0
        self.total_tensorboard = 0
        self.tensorboardx_name = None

    def commandline_accuracy_init(self):
        """
        Reset the variables used for command line accuracy display
        """
        self.correct_commandline = 0
        self.total_commandline = 0

    def tensorboardx_accuracy_init(self, tensorboardx_name="Undefined"):
        """
        Reset the variables used for tensorboardX accuracy display
        This method will record the timestamp of init and used as an id for tensorboard runs
        :param tensorboardx_name: name for tag displayed in tensorboard
        """
        self.start_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
        self.writer = SummaryWriter(self.__LOG_ADDR + self.start_time + "_accuracy")

        self.correct_tensorboard = 0
        self.total_tensorboard = 0
        self.tensorboardx_name = tensorboardx_name
        print('accuracy monitor run timestamp: ' + self.start_time)


    def commandline_accuracy_monitor(self, labels=torch.tensor([])
                                     , predicted=torch.tensor([]), display=False
                                     , tracking_mode=False, epoch=0
                                     , idx=0, dataloader=None, interval=0):
        """
        1. Method for display accuracy on command line
        2. We can use it in training for accuracy check with tracking_mode=True
        3. We can also use it for test, display=False for accumulate number of correct
            display=True for output accuracy
        :param labels: labels in each batch, is a 1d tensor
        :param predicted: prediction in each batch, is a 1d tensor
        :param display: boolean for whether we display accuracy using existing statistics
        :param tracking_mode: boolean for whether we track accuracy during training
        :param epoch: the current epoch during training
        :param idx: the current idx in current epoch
        :param dataloader: the dataloader we used in training
        :param interval: determine display accuracy when idx reach the value of interval
        """
        self.correct_commandline += (predicted == labels).sum().item()
        self.total_commandline += labels.size(0)

        if tracking_mode and idx % interval == interval - 1:
            niter = epoch * len(dataloader) + idx
            print("[{:d}, {:d}] accuracy: {:.3f}%".format(
                epoch, niter, 100 * self.correct_commandline / self.total_commandline))
            self.correct_commandline = 0
            self.total_commandline = 0

        if display:
            print("accuracy : {:.2f}%".format(
                100 * self.correct_commandline / self.total_commandline))
            self.correct_commandline = 0
            self.total_commandline = 0


    def tensorboardx_accuracy_monitor(self, labels=torch.tensor([])
                                      , predicted=torch.tensor([]), display=False
                                      , tracking_mode=False, epoch=0
                                      , idx=0, dataloader=None, interval=0):
        """
        1. Method for display accuracy on tensorboardX
        2. We can use it in training for accuracy check with tracking_mode=True
        3. We can also use it for test, display=False for accumulate number of correct
            display=True for output accuracy
        4. When we use it in training, the output will be scalar in tensorboard
            When we use it in test, the output will be text in tensorboard for the final accuracy
        5. The tag of the graph will be related to the tensorbardx_name set in init method
        6. The runs of this graph will be set in init method with current timestamp
        :param labels: labels in each batch, is a 1d tensor
        :param predicted: prediction in each batch, is a 1d tensor
        :param display: boolean for whether we display accuracy using existing statistics
        :param tracking_mode: boolean for whether we track accuracy during training
        :param epoch: the current epoch during training
        :param idx: the current idx in current epoch
        :param dataloader: the dataloader we used in training
        :param interval: determine display accuracy when idx reach the value of interval
        """
        self.correct_tensorboard += (predicted == labels).sum().item()
        self.total_tensorboard += labels.size(0)

        if tracking_mode and idx % interval == interval - 1:
            niter = epoch * len(dataloader) + idx
            self.writer.add_scalar(self.tensorboardx_name + "_Accuracy"
                                   , 100 * self.correct_tensorboard / self.total_tensorboard, niter)
            self.correct_tensorboard = 0
            self.total_tensorboard = 0
            self.writer.flush()

        if display:
            self.writer.add_text(self.tensorboardx_name
                                 + "_Accuracy_"
                                 + time.strftime('%Y%m%d%H%M%S', time.localtime())
                                 , "Accuracy {:.2f}".format(100 * self.correct_tensorboard
                                                            / self.total_tensorboard)
                                 + r"%")
            self.correct_tensorboard = 0
            self.total_tensorboard = 0
            self.writer.flush()
