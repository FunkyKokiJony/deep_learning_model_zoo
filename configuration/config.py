"""

"""
from configparser import ConfigParser

from configuration.constants import CONFIG_ADDR


class Config:
    def __init__(self):
        self.cfg = ConfigParser()
        self.cfg.read(CONFIG_ADDR)

    def get(self, section, key):
        return self.cfg.get(section, key)