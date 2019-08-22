"""
This module is for the definition of project level constant
"""
from pathlib import Path

PROJECT_ADDR = str(Path(__file__).parent.parent)

RESOURCES_ADDR = PROJECT_ADDR + "/resources/"

CONFIG_ADDR = PROJECT_ADDR + "/config.ini"

class TensorboardConfig:
    SECTION = "tensorboard"
    LOGDIR = "logdir"

class TrainingConfig:
    SECTION = "training"
    CHECKPOINT_DIR = "checkpoint_dir"

class AccuracyStats:
    """
    This is the constants for AccuracyCallback
    """
    LABELS = "labels"
    PREDICTS = "predicts"

class MonitorMode:
    """
    This is the constants for Monitor class
    """
    TRACK = "track"
    DISPLAY = "display"
    EVAL = "eval"

class LossStats:
    """
    This is the constants for LossCallback
    """
    LOSS = "loss"
