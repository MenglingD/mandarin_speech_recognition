""" logging utility """

import logging


def set_logger(path, level):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(level)
    fh = logging.FileHandler(path, mode='w')
    logger.addHandler(fh)
