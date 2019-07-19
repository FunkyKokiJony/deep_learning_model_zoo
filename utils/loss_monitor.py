from configuration import settings
from tensorboardX import SummaryWriter
import time

class LossMonitor:
    __LOG_ADDR = settings.RESOURCES_ADDR + '/log/'

    def __init__(self):
        pass

    def commandLineLossInit(self):
        self.runningLossCommandLine = 0
        self.totalCommandLine = 0

    def tensorboardxLossInit(self, tensorboardxName="Undefined"):
        self.startTime = time.strftime('%Y%m%d%H%M%S', time.localtime())
        self.writer = SummaryWriter(self.__LOG_ADDR + self.startTime + "_loss")

        self.runningLossTensorBoard = 0
        self.totalTensorBoard = 0
        self.tensorboardxName = tensorboardxName
        print('loss monitor run timestamp: ' + self.startTime)

    def commandLineLossMonitor(self, epoch, idx, loss, dataloader, batch=2000, display=False):
        self.runningLossCommandLine += loss
        self.totalCommandLine += 1
        if idx % batch == batch - 1 or display:
            niter = epoch * len(dataloader) + idx
            print("[{:d}, {:d}] loss: {:.3f}".format(
                epoch, idx, self.runningLossCommandLine/self.totalCommandLine))
            self.runningLossCommandLine = 0
            self.totalCommandLine = 0

    def tensorboardxLossMonitor(self, epoch, idx, loss, dataloader, batch=10, display=False):
        self.runningLossTensorBoard += loss
        self.totalTensorBoard += 1
        if idx % batch == batch - 1 or display:
            niter = epoch * len(dataloader) + idx
            self.writer.add_scalar(self.tensorboardxName + "_Loss"
                                   , self.runningLossTensorBoard/self.totalTensorBoard, niter)
            self.runningLossTensorBoard = 0
            self.totalTensorBoard = 0
            self.writer.flush()
