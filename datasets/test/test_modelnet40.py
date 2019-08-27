"""

"""
import logging

from datasets.modelnet40 import ModelNet40


def test_modelnet40():
    modelnet40 = ModelNet40(2048)
    logging.info(modelnet40.point_list[1].shape)
    logging.info(modelnet40.label_list[0].shape)