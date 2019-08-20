"""
This module is for the definition of project level constant
"""
from pathlib import Path

PROJECT_ADDR = str(Path(__file__).parent.parent)

RESOURCES_ADDR = PROJECT_ADDR + "/resources/"

class AccuracyStats:
    LABELS = "labels"
    PREDICTS = "predicts"

class MonitorMode:
    TRACK = "track"
    EVAL = "eval"

class LossStats:
    LOSS = "loss"
