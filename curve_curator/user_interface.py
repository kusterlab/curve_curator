# user_interface.py
# Functions related to cmd line outputs for better interaction with the user.
#
# Florian P. Bayer - 2024
#

import os
import sys
import tomllib
import numpy as np
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

                                               Florian P. Bayer - 2024
    
    Please cite CurveCurator: 10.1038/s41467-023-43696-z"""
    print(TerminalFormatting.OKCYAN + msg + TerminalFormatting.ENDC, end='\n\n')


def breakline():
    """
    prints a break line to the terminal
    """
    msg = 70 * '#'
    print(TerminalFormatting.OKCYAN + msg + TerminalFormatting.ENDC, end='\n\n')


def doneline(end='\n\n'):
    """
    prints a done message to the terminal
    """
    msg = "\n" + 32 * '#' + ' DONE ' + 32 * '#' + "\n"
    msg = f'{TerminalFormatting.OKGREEN}{msg}{TerminalFormatting.ENDC}'
    print(msg, end=end)


def message(msg, terminal_only=False, end='\n\n'):
    """
    prints the message to the terminal and logging file
    """
    if LOGGER and not terminal_only:
        LOGGER.info(msg)
    else:
        print(msg, end=end)


def warning(msg, end='\n\n'):
    """
    prints a warning message to the terminal and logging file
    """
    msg = f'{TerminalFormatting.WARNING}{msg}{TerminalFormatting.ENDC}'
    if LOGGER:
        LOGGER.warning(msg)
    else:
        print(msg, end=end)


def error(msg, end='\n\n\n\n'):
    """
    prints a error message to the terminal and logging file
    """
    error_line = "\n" + 32 * '#' + ' ERROR ' + 31 * '#' + "\n\n"
    msg = f'{TerminalFormatting.FAIL}{error_line}{msg}{TerminalFormatting.ENDC}'
    if LOGGER:
        LOGGER.error(msg)
    else:
        print(msg, end=end)


def is_toml_file(file_path):
    """
    Checks if the file_path leads to a toml file
    """
    return os.path.splitext(file_path)[-1].lower().endswith('.toml')


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


def update_toml_paths(config):
    """
    update_toml_paths(config)

    Update all relative paths from the toml file to absolute paths based on the cwd of the script.
    """
    # Define the absolute path
    config_path = config['__file__']['Path']
    abs_folder_path = os.path.abspath(os.path.dirname(config_path))

    # Update all paths from the toml Path section if they are not yet updated to an absolute path
    for key, path in config['Paths'].items():
        if not os.path.isabs(path):
            config['Paths'][key] = str(os.path.join(abs_folder_path, path))
            if sys.platform.startswith('linux'):
                config['Paths'][key] = str(config['Paths'][key].replace("\\", "/"))


def check_toml_params(config):
    """
    Check the toml file for validity and warn the user if unexpected values are detected or raise an value error
    if implausible value are detected.

    Parameters
    ----------
    config : toml object (Dict of dicts)
        toml file with all the parameters.

    Returns
    -------
    None
    """
    try:
        #
        # ['Meta']
        #
        for value in ['id', 'condition', 'description', 'treatment_time']:
            if not config['Meta'][value]:
                error(f"Error: [Meta] {value} is empty.")
                raise ValueError(f"[Meta] {value}")

        #
        # ['Experiment']
        #
        experiments = np.array(config['Experiment']['experiments'])
        control_experiment = np.array([config['Experiment']['control_experiment']]).flatten()
        doses = np.array(config['Experiment']['doses'])
        dose_scale = config['Experiment']['dose_scale']
        dose_unit = config['Experiment']['dose_unit']

        if len(experiments) < 2 or len(doses) < 2:
            error("Error: [Experiment] 'experiments' and [Experiment] 'doses' need at least length 2.")
            raise ValueError("[Experiment] 'experiments' & 'doses' length")
        if len(experiments) != len(doses):
            error("Error: [Experiment] 'experiments' and [Experiment] 'doses' do no correspond in length.")
            raise ValueError("[Experiment] 'experiments' & 'doses' length")
        if len(set(control_experiment) - set(experiments)) > 0:
            error("Error: [Experiment] at least one 'control_experiment' is not in [Experiment] 'experiments'.")
            raise ValueError("[Experiment] 'experiments'")
        if not dose_scale:
            error("Error: [Experiment] 'dose_scale' is empty.")
            raise ValueError("[Experiment] 'dose_scale'")
        if not dose_unit:
            error("Error: [Experiment] 'dose_unit' is empty.")
            raise ValueError("[Experiment] 'dose_unit'")

        #
        # ['Paths']
        #
        for value in ['input_file', 'curves_file', 'normalization_file', 'mad_file', 'dashboard']:
            toml_parameter = config['Paths'].get(value)
            if toml_parameter is None:
                continue
            if not toml_parameter:
                error(f"Error: [Paths] '{value}' is empty. Wither provide a valid file name or remove {value} from toml file and use default value.")
                raise ValueError(f"[Paths] '{value}'")

        #
        # ['Processing']
        #

        # booleans
        for value in ['imputation', 'normalization']:
            toml_parameter = config['Processing'].get(value)
            if toml_parameter is None:
                continue
            if type(toml_parameter) is not bool:
                error(f"Error: [Processing] {value} must be true or false.")
                raise ValueError(f"[Processing] {value}")

        # floats
        for value in ['front', 'slope']:
            toml_parameter = config.get('Curve Fit', {}).get(value)
            if toml_parameter is None:
                continue
            if type(toml_parameter) is not float:
                error(f"Error: [Curve Fit] {value} must be a float.")
                raise ValueError(f"[Curve Fit] {value}")

        # alpha
        curve_alpha = config['F Statistic'].get('alpha')
        if not (0.0 < curve_alpha <= 1.0):
            error("Error: [F Statistic] 'alpha' must be between 0.0 and 1.0.")
            raise ValueError("[F Statistic] 'alpha'")

        # curve fold change limit
        curve_fclim = config['F Statistic'].get('fc_lim')
        if not (0.0 <= curve_fclim):
            error("Error: [F Statistic] 'fc_lim' must be  >= 0.0.")
            raise ValueError("[Processing] 'fc_lim'")

    except ValueError as VE:
        warning("Please fix your toml file and try again.\n")
        raise VE


def load_toml(path, random_mode=False):
    """
    Load the toml file and ignore File not found error in random mode.
    """
    message(' * Reading parameter file of experiment.')

    # Check & load the toml file, and add the path variable to toml file
    check_path(path)
    with open(path, "rb") as f:
        config = tomllib.load(f)
    config['__file__'] = {'Path': path}

    # Check the parameter file values
    try:
        check_toml_params(config)
    except ValueError as parameter_error:
            error('Issue(s) with the toml file found!! Please check.', end='\n')
            error(parameter_error)
            exit()

    # Check the input file exists
    try:
        update_toml_paths(config)
        check_path(config['Paths']['input_file'])
    except FileNotFoundError:
        if not random_mode:
            error('Issue(s) with the toml file found!! The input file cannot be found! Please check.')
            exit()
    return config


def verify_columns_exist(df, columns):
    """
    Checks if all columns are present in the data frame. Else raises Error message and end the program.
    """
    for col in columns:
        if col not in df:
            error(f'The column "{col}" was not found in your data. Please fix your input data or the TOML file.', end='\n')
            exit()


def set_default_values(config):
    """
    Sets default values for optional parameters of the pipeline when the user didn't specify it.
    """
    experiments = np.array(config['Experiment']['experiments']).flatten()
    control_experiments = np.array([config['Experiment']['control_experiment']]).flatten()
    doses = np.array([config['Experiment']['doses']]).flatten()

    # Experiment
    exp_params = config['Experiment']
    exp_params['experiments'] = experiments
    exp_params['control_experiment'] = control_experiments
    exp_params['doses'] = doses
    exp_params['dose_scale'] = float(exp_params.get('dose_scale', 1e0))
    config['Experiment'] = exp_params

    # Paths
    path_params = config.get('Paths', {})
    path_params['curves_file'] = str(path_params.get('curves_file', './curves.txt'))
    path_params['decoys_file'] = str(path_params.get('decoys_file', './decoys.txt'))
    path_params['fdr_file'] = str(path_params.get('fdr_file', './fdr.txt'))
    path_params['mad_file'] = str(path_params.get('mad_file', './mad.txt'))
    path_params['dashboard'] = str(path_params.get('dashboard', './dashboard.html'))
    config['Paths'] = path_params
    update_toml_paths(config)

    # Processing
    proc_params = config.get('Processing', {})
    proc_params['available_cores'] = int(proc_params.get('available_cores', 1))  # optional int
    proc_params['imputation'] = bool(proc_params.get('imputation', False))
    proc_params['imputation_pct'] = float(proc_params.get('imputation_pct', 0.005))
    proc_params['normalization'] = bool(proc_params.get('normalization', False))
    proc_params['max_missing'] = int(proc_params.get('max_missing', len(experiments)))
    proc_params['ratio_range'] = proc_params.get('ratio_range', None)
    config['Processing'] = proc_params

    # Curve Fit
    fit_params = config.get('Curve Fit', {})
    fit_params['weights'] = fit_params.get('weights', None)  # list of weights
    if isinstance(fit_params['weights'], list):
        fit_params['weights'] = np.array(fit_params['weights'])
    fit_params['interpolation'] = bool(fit_params.get('interpolation', False))
    fit_params['type'] = str(fit_params.get('type', 'OLS'))  # can be: 'OLS', 'MLE',
    fit_params['speed'] = str(fit_params.get('speed', 'standard'))  # can be: 'fast', 'standard', 'extensive', 'basinhopping'
    fit_params['max_iterations'] = int(fit_params.get('max_iterations', 100 * len(experiments)))  # optional int
    fit_params['control_fold_change'] = bool(fit_params.get('control_fold_change', False))  # to fix the fold change calculation to control 1
    config['Curve Fit'] = fit_params

    # F statistic
    f_statistic_params = config.get('F Statistic', {})
    f_statistic_params['decoy_ratio'] = float(f_statistic_params.get('decoy_ratio', 1.0))
    f_statistic_params['optimized_dofs'] = bool(f_statistic_params.get('optimized_dofs', True))  # can be: True, False
    f_statistic_params['loc'] = float(f_statistic_params.get('loc', 0.12))
    f_statistic_params['scale'] = float(f_statistic_params.get('scale', 1))
    f_statistic_params['two_sided'] = bool(f_statistic_params.get('two_sided', False))  # F-test is a one-sided test.
    f_statistic_params['quality_min'] = float(f_statistic_params.get('quality_min', -np.inf))  # Have no minimal filter by default.
    f_statistic_params['mtc_method'] = str(f_statistic_params.get('mtc_method', 'sam'))
    f_statistic_params['not_rmse_limit'] = float(f_statistic_params.get('not_rmse_limit', 0.1))
    f_statistic_params['not_p_limit'] = float(f_statistic_params.get('not_p_limit', np.inf))  # Default is no max p-value filter
    config['F Statistic'] = f_statistic_params

    # Dashboard
    bokeh_params = config.get('Dashboard', {})
    bokeh_params['backend'] = str(bokeh_params.get('backend', 'webgl'))
    config['Dashboard'] = bokeh_params

    return config
