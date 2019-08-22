"""

"""
from configparser import ConfigParser

from configuration.constants import CONFIG_ADDR
import os


class Config:
    def __init__(self):
        self.cfg = ConfigParser()
        self.cfg.set("DEFAULT", "home", os.path.expanduser("~"))
        self.cfg.read(CONFIG_ADDR)

    def get(self, section, key):
        return self.cfg.get(section, key)