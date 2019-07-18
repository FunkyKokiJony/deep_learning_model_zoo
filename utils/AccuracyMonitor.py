from configuration import settings
from tensorboardX import SummaryWriter
import torch
import time

class AccuracyMonitor:
    __LOG_ADDR = settings.RESOURCES_ADDR + '/log/'

    def __init__(self):
        pass

    def commandLineAccuracyInit(self):
        self.correctCommandLine = 0
        self.totalCommandLine = 0

    def tensorboardxAccuracyInit(self, tensorboardxName="Undefined"):
        self.startTime = time.strftime('%Y%m%d%H%M%S', time.localtime())
        self.writer = SummaryWriter(self.__LOG_ADDR + self.startTime + "_accuracy")

        self.correctTensorBoard = 0
        self.totalTensorBoard = 0
        self.tensorboardxName = tensorboardxName
        print('accuracy monitor run timestamp: ' + self.startTime)


    def commandLineAccuracyMonitor(self, labels=torch.tensor([]), predicted=torch.tensor([]), display=False, tracking_mode=False, epoch=0 , idx=0, dataloader=None, batch=0):
        self.correctCommandLine += (predicted == labels).sum().item()
        self.totalCommandLine += labels.size(0)

        if tracking_mode and idx % batch == batch - 1:
            niter = epoch * len(dataloader) + idx
            print("[{:d}, {:d}] accuracy: {:.3f}%".format(epoch, niter, 100 * self.correctCommandLine / self.totalCommandLine))
            self.correctCommandLine = 0
            self.totalCommandLine = 0

        if display:
            print("accuracy : {:.2f}%".format(100 * self.correctCommandLine / self.totalCommandLine))
            self.correctCommandLine = 0
            self.totalCommandLine = 0


    def tensorboardxAccuracyMonitor(self, labels=torch.tensor([]), predicted=torch.tensor([]), display=False, tracking_mode=False, epoch=0 , idx=0, dataloader=None, batch=0):
        self.correctTensorBoard += (predicted == labels).sum().item()
        self.totalTensorBoard += labels.size(0)

        if tracking_mode and idx % batch == batch - 1:
            niter = epoch * len(dataloader) + idx
            self.writer.add_scalar(self.tensorboardxName + "_Accuracy", 100 * self.correctTensorBoard / self.totalTensorBoard, niter)
            self.correctTensorBoard = 0
            self.totalTensorBoard = 0
            self.writer.flush()

        if display:
            self.writer.add_text(self.tensorboardxName + "_Accuracy_" + time.strftime('%Y%m%d%H%M%S', time.localtime())
                                 , "Accuracy {:.2f}".format(100 * self.correctTensorBoard / self.totalTensorBoard) + r"%")
            self.correctTensorBoard = 0
            self.totalTensorBoard = 0
            self.writer.flush()
