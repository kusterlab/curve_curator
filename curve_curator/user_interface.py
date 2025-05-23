# user_interface.py
# Functions related to cmd line outputs for better interaction with the user.
#
# Florian P. Bayer - 2025
#

import os
import sys
import logging
import time
from pathlib import Path
os.system("")
LOGGER = None


# ANSI escape sequences for Terminal Text Formatting.
# Usage is f"{START TOKEN} ... {END TOKEN}"
class TerminalFormatting:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def setup_logger(directory, name):
    """
    Set up a logger that prints all output both to the command line
    and to a file in the directory of the configuration TOML file.
    """
    global LOGGER
    LOGGER = logging.getLogger(str(name))
    LOGGER.setLevel(logging.DEBUG)

    file_logger = logging.FileHandler(directory / Path('curveCurator.log'), mode='w')
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    formatter.converter = time.gmtime
    file_logger.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
    formatter = logging.Formatter("%(message)s \n")
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(message)s \n\n")
    stderr_handler.setFormatter(formatter)

    LOGGER.addHandler(file_logger)
    LOGGER.addHandler(stdout_handler)
    LOGGER.addHandler(stderr_handler)

    # Override the system's excepthook so uncaught exceptions are also logged
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        LOGGER.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception
    return LOGGER


def welcome():
    """
    prints a welcome message to the terminal
    """
    msg = \
    """
    #########              #################              #########
             #             # Curve Curator #             #
              #            #################            #
               0 pEC50                           pEC50 0
                #                                     #
                 #                                   #
                  #########                 #########

                                               Florian P. Bayer - 2025

    Please cite CurveCurator: 10.1038/s41467-023-43696-z"""
    print(TerminalFormatting.OKCYAN + msg + TerminalFormatting.ENDC, end='\n\n')


def breakline():
    """
    prints a break line to the terminal
    """
    msg = 70 * '#'
    print(TerminalFormatting.OKCYAN + msg + TerminalFormatting.ENDC, end='\n\n')
    return msg


def doneline(end='\n\n'):
    """
    prints a done message to the terminal
    """
    msg = "\n" + 32 * '#' + ' DONE ' + 32 * '#' + "\n"
    msg = f'{TerminalFormatting.OKGREEN}{msg}{TerminalFormatting.ENDC}'
    print(msg, end=end)
    return msg


def errorline(end='\n\n'):
    """
    prints a error line to the terminal
    """
    msg = "\n" + 32 * '#' + ' ERROR ' + 31 * '#' + "\n"
    msg = f'{TerminalFormatting.FAIL}{msg}{TerminalFormatting.ENDC}'
    print(msg, end=end)
    return msg


def message(msg, terminal_only=False, end='\n\n'):
    """
    prints the message to the terminal and logging file
    """
    if LOGGER and not terminal_only:
        LOGGER.info(msg)
    else:
        print(msg, end=end)
    return msg


def warning(msg, end='\n\n'):
    """
    prints a warning message to the terminal and logging file
    """
    msg = f'{TerminalFormatting.WARNING}{msg}{TerminalFormatting.ENDC}'
    if LOGGER:
        LOGGER.warning(msg)
    else:
        print(msg, end=end)
    return msg


def error(msg, end='\n\n'):
    """
    prints a error message to the terminal and logging file
    """
    errorline()
    msg = f'{TerminalFormatting.FAIL}{msg}{TerminalFormatting.ENDC}'
    if LOGGER:
        LOGGER.error(msg)
    else:
        print(msg, end=end)
    return msg


def check_path(path, is_dir=False):
    """
    check_path(path, is_dir=False)

    Checks if a path exists and is accessible. If not an error is thrown.
    The dir_flag checks if the path is supposed to be a directory or file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'Path "{path}" does not exist.')
    if not os.access(path, os.R_OK):
        raise PermissionError(f'File at "{path}" cannot be opened. Try to close it elsewhere.')
    if not is_dir and os.path.isdir(path):
        raise ValueError(f'Path "{path}" is a folder, not a file.')
    if is_dir and not os.path.isdir(path):
        raise ValueError(f'Path "{path}" is a file, not a folder.')


def verify_columns_exist(df, columns):
    """
    Checks if all columns are present in the data frame. Else raises Error message and end the program.
    """
    for col in columns:
        if col not in df:
            error(f'The column "{col}" was not found in your data. Please fix your input data or the TOML file.', end='\n')
            exit()
