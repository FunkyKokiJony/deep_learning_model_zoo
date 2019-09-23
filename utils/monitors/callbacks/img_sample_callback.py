"""

"""
import logging

from configuration.constants import ImgSampleStats
from utils.monitors.callbacks.stats_callback import StatsCallback


class ImgSampleCallBack(StatsCallback):
    """
    For the display of this callback, should put key value pair into the stats dict
    """
    def __init__(self, interval):
        self.interval = interval

    def __call__(self, monitor, mode, idx, stats_dict=dict()):
        _tensors_dict = stats_dict.get(ImgSampleStats.IMG_SAMPLES)

        if _tensors_dict is None:
            logging.error("Key [{}] is not set".format(ImgSampleStats.IMG_SAMPLES))
            return

        if (idx % self.interval) == 0 and idx != 0:
            for _tag, _tensor in _tensors_dict.items():
                monitor.add_img_samples(_tag, _tensor, idx)

    def get_name(self):
        return "img_sample"
