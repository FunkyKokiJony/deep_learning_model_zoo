"""

"""
import logging
import os
import time

import torch.nn as nn
import torch
from torch.optim.optimizer import Optimizer

from configuration.config import Config
from configuration.constants import TrainingConfig


class CheckpointHandler:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.cfg = Config()
        self.checkpoint_dir = self.cfg.get(TrainingConfig.SECTION
                                           , TrainingConfig.CHECKPOINT_DIR)
        if model is None or not isinstance(model, nn.Module):
            logging.error("Can not save checkpoint because " +
                          "model is invalid")
        if self.optimizer is None or not isinstance(optimizer, Optimizer):
            logging.info("Can not save optimizer stats")

    def save_checkpoint(self, filename="checkpoint", epoch=None, idx=None, loss=None):
        if self.model is not None:
            filename = filename + "_" + type(self.model).__name__

        filename = "{}/{}_{}.pth.tar".format(self.checkpoint_dir
                                             , filename
                                             , time.strftime('%Y%m%d%H%M%S', time.localtime()))
        stats = dict()
        if epoch is not None:
            stats["epoch"] = epoch

        if idx is not None:
            stats["idx"] = idx

        if loss is not None:
            stats["loss"] = loss

        if self.model is not None and isinstance(self.model, nn.Module):
            stats["model_state_dict"] = self.model.state_dict()

        if self.optimizer is not None and isinstance(self.optimizer, Optimizer):
            stats["optimizer_state_dict"] = self.optimizer.state_dict()

        torch.save(stats, filename)

    def load_checkpoint(self, filename="checkpoint.pth.tar"):
        if os.path.isfile(filename):
            logging.info("=== Loading from checkpoint {} ===".format(filename))
            checkpoint = torch.load(filename)
            stats = {}

            if "epoch" in checkpoint:
                stats["epoch"] = checkpoint["epoch"]

            if "idx" in checkpoint:
                stats["idx"] = checkpoint["idx"]

            if "loss" in checkpoint:
                stats["loss"] = checkpoint["loss"]

            if self.model is not None and isinstance(self.model, nn.Module) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])

            if self.optimizer is not None and isinstance(self.optimizer, Optimizer) and "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            logging.info("=== Done ===")
            return stats

        else:
            logging.warning("Checkpoint {} not found".format(filename))
            return None




