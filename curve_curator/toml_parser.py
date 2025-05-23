# toml_parser.py
# Functions related to the CurveCurator config and parsing .toml files.
#
# Florian P. Bayer - 2025
#

import os
import sys
import tomllib
import numpy as np

from . import user_interface as ui

#
# TOML REFERENCE
# All valid toml file arguments sorted by sections
#
REFERENCE = {
    'Meta': ['id', 'condition', 'description', 'treatment_time'],
    'Experiment': ['experiments', 'doses', 'dose_scale', 'dose_unit', 'control_experiment', 'measurement_type', 'data_type', 'search_engine', 'search_engine_version'],
    'Paths': ['input_file', 'curves_file', 'decoys_file', 'fdr_file', 'normalization_file', 'mad_file', 'dashboard'],
    'Processing': ['available_cores', 'max_missing', 'max_imputation', 'imputation', 'normalization', 'ratio_range'],
    'Curve Fit': ['front', 'slope', 'back', 'weights', 'interpolation', 'type', 'speed', 'max_iterations', 'control_fold_change', 'interpolation'],
    'F Statistic': ['alpha', 'fc_lim', 'optimized_dofs', 'loc', 'scale', 'dfn', 'dfd', 'quality_min', 'mtc_method', 'not_rmse_limit', 'not_p_limit', 'decoy_ratio', 'pEC50_filter'],
    'Dashboard': ['backend'],
}


def is_toml_file(file_path):
    """
    Checks if the file_path leads to a toml file
    """
    return os.path.splitext(file_path)[-1].lower().endswith('.toml')


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


def assert_section_exits(section_name, config):
    """
    Asserts that section exits in given config file. If not, raise a ValueError.

    Parameters
    ----------
    section_name : str
        name of the section.
    config : toml object (Dict of dicts)
        toml file with all the parameters.

    Returns
    -------
    None
    """
    if not section_name in config:
        ui.error(f"TOML Error: The [{section_name}] section is missing in the toml file. Please add.")
        raise ValueError(f"[{section_name}] section is missing in the toml file.")


def check_for_unknown_keys(section_name, config):
    """
    Check for unknown key-value pairs in a specific section in the config file.
    If unknown key is found, trigger a user warning.
    If section name is not present in config, ignore.

    Parameters
    ----------
    section_name : str
        name of the section.
    config : toml object (Dict of dicts)
        toml file with all the parameters.

    Returns
    -------
    None
    """
    global REFERENCE
    for key in config.get(section_name, dict()):
        if key not in REFERENCE[section_name]:
            ui.warning(f"TOML Warning: In [{section_name}]: the key '{key}' is unknown. Please double check !")


def check_for_required_keys(section_name, config, key_list):
    """
    Check for required key-value pairs in a specific section in the config file.
    If required key from key_list is not present in the specific config section, raise a ValueError.

    Parameters
    ----------
    section_name : str
        name of the section.
    config : toml object (Dict of dicts)
        toml file with all the parameters.
    key_list : list(key1, key2, ...)
        list of keys to check

    Returns
    -------
    None
    """
    for key in key_list:
        if key not in config[section_name]:
            ui.error(f"TOML Error: In [{section_name}]: the required key '{key}' is missing. Please add.")
            raise ValueError(f"[{section_name}] {key}")


def check_for_correct_values(section_name, config, key_list, requirement):
    """
    Check for correct value requirements in a specific section in the config file for keys in key_list.
    If key from key_list does not meet requirements, raise a ValueError.
    If key is not present in the section, ignore.

    Parameters
    ----------
    section_name : str
        name of the section.
    config : toml object (Dict of dicts)
        toml file with all the parameters.
    key_list : list(key1, key2, ...)
        list of keys to check
    requirement : function
        This requirement function is tested against all keys in the key_list.
        If the requirement function evaluates TRUE -> pass.
        If the requirement function evaluates FALSE -> trigger ValueError.

    Returns
    -------
    None
    """
    for key in key_list:
        value = config.get(section_name, dict()).get(key)
        if value is None:
            continue
        if not requirement(value):
            ui.error(f"TOML Error: In [{section_name}]: the key-value pair '{key} = {value}' does not meet the specifications. Please fix.")
            raise ValueError(f"[{section_name}] {key}")


def check_correct_experiment_setup(config):
    """
    Check that the experiment is correctly setup.
    If not raise a ValueError and warn the user.

    Parameters
    ----------
    config : toml object (Dict of dicts)
        toml file with all the parameters.

    Returns
    -------
    None
    """
    # Load values
    experiments = np.array(config['Experiment']['experiments'])
    control_experiment = np.array([config['Experiment']['control_experiment']]).flatten()
    doses = np.array(config['Experiment']['doses'])
    dose_scale = config['Experiment']['dose_scale']
    dose_unit = config['Experiment']['dose_unit']

    # Perform the tests
    if (len(experiments) < 4) or (len(doses) < 4):
        ui.error("TOML Error: [Experiment] 'experiments' and [Experiment] 'doses' need at least length 4.")
        raise ValueError("[Experiment] 'experiments' & 'doses' length")
    if len(experiments) != len(doses):
        ui.error("TOML Error: [Experiment] 'experiments' and [Experiment] 'doses' do no correspond in length.")
        raise ValueError("[Experiment] 'experiments' & 'doses' length")
    if len(set(control_experiment) - set(experiments)) > 0:
        ui.error("TOML Error: [Experiment] at least one 'control_experiment' is not in [Experiment] 'experiments'.")
        raise ValueError("[Experiment] 'experiments'")
    if not dose_scale:
        ui.error("TOML Error: [Experiment] 'dose_scale' is empty.")
        raise ValueError("[Experiment] 'dose_scale'")
    if not dose_unit:
        ui.error("TOML Error: [Experiment] 'dose_unit' is empty.")
        raise ValueError("[Experiment] 'dose_unit'")
    if len(experiments) != len(set(experiments)):
        ui.error("TOML Error: [Experiment] 'experiments' contains duplicates. Make sure experiment names are unique.")
        raise ValueError("[Experiment] 'experiments' contains duplicates.")
    if len(control_experiment) != len(set(control_experiment)):
        ui.error("TOML Error: [Experiment] 'control_experiment' contains duplicates. Make sure experiment names are unique.")
        raise ValueError("[Experiment] 'control_experiment' contains duplicates.")


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
        assert_section_exits('Meta', config)
        check_for_unknown_keys('Meta', config)
        required_keys = ['id', 'condition', 'description', 'treatment_time']
        check_for_required_keys('Meta', config, required_keys)

        #
        # ['Experiment']
        #
        assert_section_exits('Experiment', config)
        check_for_unknown_keys('Experiment', config)
        required_keys = ['experiments', 'control_experiment', 'doses', 'dose_scale', 'dose_unit']
        check_for_required_keys('Experiment', config, required_keys)
        check_correct_experiment_setup(config)

        #
        # ['Paths']
        #
        assert_section_exits('Paths', config)
        check_for_unknown_keys('Paths', config)
        required_keys = ['input_file']
        check_for_required_keys('Paths', config, required_keys)
        check_for_correct_values('Paths', config, REFERENCE['Paths'], requirement=lambda v: len(v) > 0)

        #
        # ['Processing']
        #
        # ['Processing'] is optional and no field is required.
        check_for_unknown_keys('Processing', config)
        required_keys = []
        check_for_required_keys('Processing', config, required_keys)
        keys = ['imputation', 'normalization']
        check_for_correct_values('Processing', config, keys, requirement=lambda v: type(v) is bool)
        keys = ['ratio_range']
        check_for_correct_values('Processing', config, keys, requirement=lambda v: len(v) == 2)

        #
        # ['Curve Fit']
        #
        # ['Curve Fit'] is optional and no field is required.
        check_for_unknown_keys('Curve Fit', config)
        required_keys = []
        check_for_required_keys('Curve Fit', config, required_keys)
        keys = ['control_fold_change', 'interpolation']
        check_for_correct_values('Curve Fit', config, keys, requirement=lambda v: type(v) is bool)
        keys = ['front', 'slope', 'back']
        check_for_correct_values('Curve Fit', config, keys, requirement=lambda v: type(v) is float)
        keys = ['type']
        check_for_correct_values('Curve Fit', config, keys, requirement=lambda v: v in {'OLS', 'MLE'})
        keys = ['speed']
        check_for_correct_values('Curve Fit', config, keys, requirement=lambda v: v in {'fast', 'standard', 'exhaustive', 'basinhopping'})

        #
        # ['F Statistic']
        #
        assert_section_exits('F Statistic', config)
        check_for_unknown_keys('F Statistic', config)
        required_keys = ['alpha', 'fc_lim']
        check_for_required_keys('F Statistic', config, required_keys)

        # alpha
        curve_alpha = config['F Statistic'].get('alpha')
        if not (0.0 < curve_alpha <= 1.0):
            ui.error("TOML Error: In [F Statistic]: the 'alpha' value must be between 0.0 and 1.0.")
            raise ValueError("[F Statistic] 'alpha'")

        # curve fold change limit
        curve_fclim = config['F Statistic'].get('fc_lim')
        if not (0.0 <= curve_fclim):
            ui.error("TOML Error: In [F Statistic]: the 'fc_lim' value must be  >= 0.0.")
            raise ValueError("[F Statistic] 'fc_lim'")

        # Load list keys and test value plausibility.
        keys = ['pEC50_filter']
        check_for_correct_values('F Statistic', config, keys, requirement=lambda v: len(v) == 2)

        #
        # ['Dashboard']
        #
        # ['Dashboard'] is optional and no field is required.
        check_for_unknown_keys('Dashboard', config)
        required_keys = []
        check_for_required_keys('Dashboard', config, required_keys)
        keys = ['backend']
        check_for_correct_values('Dashboard', config, keys, requirement=lambda v: v in {'webgl', 'svg', 'canvas'})

    # Handle raised ValueErrors and report to the user.
    except ValueError as VE:
        ui.warning("Please fix your toml file and try again.")
        ui.warning("For more information, please visit the documentation on GitHub.")
        raise VE


def load_toml(path, random_mode=False):
    """
    Load the toml file and ignore File not found error in random mode.
    """
    ui.message(' * Reading parameter file of experiment.')

    # Check & load the toml file, and add the path variable to toml file
    ui.check_path(path)
    with open(path, "rb") as f:
        config = tomllib.load(f)
    config['__file__'] = {'Path': path}

    # Check the parameter file values
    try:
        check_toml_params(config)
    except ValueError as parameter_error:
            ui.error('Issue(s) with the toml file found!! Please check.', end='\n')
            ui.error(parameter_error)
            exit()

    # Check the input file exists
    try:
        update_toml_paths(config)
        ui.check_path(config['Paths']['input_file'])
    except FileNotFoundError:
        if not random_mode:
            ui.error('Issue(s) with the toml file found!! The input file cannot be found! Please check.')
            exit()
    return config


def set_default_values(config):
    """
    Sets default values for optional parameters of the pipeline when the user didn't specify it.
    """
    experiments = np.array(config['Experiment']['experiments']).flatten().astype(str)
    control_experiments = np.array([config['Experiment']['control_experiment']]).flatten().astype(str)
    doses = np.array([config['Experiment']['doses']]).flatten().astype(float)

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
    proc_params['max_imputation'] = int(proc_params.get('max_missing', proc_params['max_missing']))  # equivalent to max_missing by default
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
    f_statistic_params['pEC50_filter'] = f_statistic_params.get('pEC50_filter', [-np.inf, np.inf])  # Default is no pEC50 filter
    config['F Statistic'] = f_statistic_params

    # Dashboard
    bokeh_params = config.get('Dashboard', {})
    bokeh_params['backend'] = str(bokeh_params.get('backend', 'webgl'))
    config['Dashboard'] = bokeh_params

    return config
