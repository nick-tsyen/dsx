import logging

"""
This module sets up a logger for the Jupyter Notebook.
It will log to both the console and a file.
"""

logger = logging.getLogger('__name__')

def set_logging(logpath:str = "test.log", level=logging.DEBUG, debug:bool=True):
    # create console handler and set level to debug
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # create file handler and set level to debug
    fileHandler = logging.FileHandler(logpath)
    fileHandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # add handlers to logger
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    # set log level for development
    logger.setLevel(logging.DEBUG) if debug else logger.setLevel(level)
    return logger

if __name__=='__main__':
    set_logging()
