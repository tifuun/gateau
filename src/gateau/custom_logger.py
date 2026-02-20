"""!
@file custom_logger.py
@brief This file contains class definitions of the custom logger objects used in gateau.
"""

import sys
import logging

from tqdm import tqdm

class CustomFormatter(logging.Formatter):
    """!
    Class for formatting of the logging from the terminal.
    Logger records date, timestamp and type of level.
    Has custom colors for different logging levels.
    """

    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;1m"
    blue = "\x1b[34;1m"
    purple = "\x1b[35;1m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s "#(%(filename)s:%(lineno)d)" 
    
    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

class CustomLogger(object):
    """!
    Class for instantiating a logger object for the terminal.
    """

    def __init__(self, owner=None):
        self.owner = "Logger" if owner is None else owner

    def __del__(self):
        del self

    def getCustomLogger(self, stdout=None):
        stdout = sys.stdout if stdout is None else stdout

        logger = logging.getLogger(self.owner)
        
        if logger.hasHandlers():
            logger.handlers = []
        
        logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(stdout)
        ch.setLevel(logging.DEBUG)

        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)

        return logger

def parallel_iterator(x, idx_thread):
    """!
    Wrapper for tqdm for for-loops contained in pooled functions.
    Makes sure that only thread 0 prints tqdm progressbar to console.
    All other threads quitly do their job.

    @param x Array over which to iterate.
    @param idx_thread Thread index, to determine if progressbar is shown.

    @returns Wrapper for tqdm if idx_thread == 0, else return x itself.
    """
    return tqdm(x, 
                ncols=100, 
                total=x.size, 
                colour="GREEN") if idx_thread == 0 else x
