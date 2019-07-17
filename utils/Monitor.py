from configuration import settings
from tensorboardX import SummaryWriter
import time

class Monitor:
    __LOG_ADDR = settings.RESOURCES_ADDR + '/log/'

    def __init__(self):
        pass

    def commandLineInit(self):
        self.runningLossCommandLine = 0

    def tensorboardxInit(self, tensorboardxName="Undefined"):
        self.startTime = time.strftime('%Y%m%d%H%M%S', time.localtime())
        self.writer = SummaryWriter(self.__LOG_ADDR + self.startTime)

        self.runningLossTensorBoard = 0
        self.tensorboardxName = tensorboardxName
        print('run timestamp: ' + self.startTime)

    def accuracyInit(self):
        self.correct = 0
        self.total = 0

    def commandLineMonitor(self, epoch, idx, loss, batch=2000):
        self.runningLossCommandLine += loss
        if idx % batch == batch - 1:
            print("[{:d}, {:d}] loss: {:.3f}".format(epoch, idx, self.runningLossCommandLine/batch))
            self.runningLossCommandLine = 0

    def tensorboardxMonitor(self, epoch, idx, loss, dataloader, batch=10):
        self.runningLossTensorBoard += loss
        if idx % batch == batch - 1:
            niter = epoch * len(dataloader) + idx
            self.writer.add_scalar(self.tensorboardxName, self.runningLossTensorBoard/batch, niter)
            self.runningLossTensorBoard = 0

    def accuracyMonitor(self, labels, predicted, display=False):
        if display:
            print("Accuracy : {:.2f}%".format(100 * self.correct / self.total))
            return

        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()

