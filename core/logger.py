"""Logger

# Example
Run command as follows to prepare dataset:


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Standard dist imports
import datetime
import logging
import os
import unittest

# Third party imports

# Project level imports

# Module level constants

class Logger(object):
    """Logger object for dataset generation and model training/testing"""

    def __init__(self, log_filename, level):
        """ Initializes logger

        Args:
            log_filename: The full path directory to log file
        """
        self.log_filename = log_filename
        dir_name = os.path.dirname(log_filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        logging.basicConfig(level=level,
                            format='',
                            datefmt='%m-%d %H:%M:S',
                            filename=log_filename,
                            filemode='w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
        logging.info('Logger initialized @ {}'.format(datetime.datetime.now()))

    @staticmethod
    def section_break(title):
        logging.info("="*30 + "   {}   ".format(title) + "="*30)

if __name__ == '__main__':
    log_filename = 'data/test.log'
    Logger(log_filename, logging.DEBUG)

    Logger.section_break(title='Arguments')
    logger = logging.getLogger('test')
    logger.info('test')


