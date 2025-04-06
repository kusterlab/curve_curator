import unittest
import io
import sys
import copy

import pandas as pd
import numpy as np
from curve_curator.toml_parser import *
import curve_curator.user_interface as ui


# Function to parse the console stream while testing for double checking output.
def parse_console(func, **kwargs):
    capturedOutput = io.StringIO()                 # Create StringIO.
    sys.stdout = capturedOutput                    # Redirect stdout.
    try:
        func(**kwargs)                             # Call function ...
    except:                                        # and continue no matter what!
        pass
    sys.stdout = sys.__stdout__                    # Reset redirect.
    return capturedOutput.getvalue()

# For console debug purpose to print out.
def log_output(path, s):
    with open(path, 'w') as f:
        f.write(s)

# Add console formatting.
def build_console_str(s, terminal_format=''):
    if format:
        s = terminal_format + s + ui.TerminalFormatting.ENDC
    return s + '\n\n'

#
# Test Area
#

class TestIsTOMLFile:
    def test_empty_imput(self):
        path = ''
        expected_result = False
        assert is_toml_file(path) is expected_result

    def test_different_file(self):
        path = 'random.txt'
        expected_result = False
        assert is_toml_file(path) is expected_result

    def test_toml_file(self):
        path = 'random.toml'
        expected_result = True
        assert is_toml_file(path) is expected_result


class TestSection(unittest.TestCase):
    def test_empty_config(self):
        config = {}
        section = 'A'
        self.assertRaises(ValueError, assert_section_exits, section, config)

    def test_section_not_present(self):
        config = {'B': {}}
        section = 'A'
        self.assertRaises(ValueError, assert_section_exits, section, config)

    def test_section_present(self):
        config = {'A':{}, 'B': {}}
        section = 'A'
        assert_section_exits(section, config) # Should not throw an error


class TestUnknownParameter(unittest.TestCase):
    # Empty config sections
    meta = {k:None for k in ['id', 'condition', 'description', 'treatment_time']}
    experiment = {k:None for k in ['experiments', 'doses', 'dose_scale', 'dose_unit', 'control_experiment', 'measurement_type', 'data_type', 'search_engine', 'search_engine_version']}
    paths = {k:None for k in ['input_file', 'curves_file', 'decoys_file', 'fdr_file', 'normalization_file', 'mad_file', 'dashboard']}
    processing = {k:None for k in ['available_cores', 'max_missing', 'max_imputation', 'imputation', 'normalization', 'ratio_range']}
    curvefit = {k:None for k in ['front', 'slope', 'back', 'weights', 'interpolation', 'type', 'speed', 'max_iterations', 'control_fold_change', 'interpolation']}
    fstatistic = {k:None for k in ['alpha', 'fc_lim', 'optimized_dofs', 'loc', 'scale', 'dfn', 'dfd', 'quality_min', 'mtc_method', 'not_rmse_limit', 'not_p_limit', 'decoy_ratio', 'pEC50_filter']}
    dashboard = {k:None for k in ['backend']}


    def test_unknown_parameter_no_error(self):
        #log_output('./test_terminal.txt', s)
        s = parse_console(check_for_unknown_keys, section_name='Meta', config={'Meta':self.meta})
        assert s == ''

        s = parse_console(check_for_unknown_keys, section_name='Experiment', config={'Experiment':self.experiment})
        assert s == ''

        s = parse_console(check_for_unknown_keys, section_name='Paths', config={'Paths':self.paths})
        assert s == ''

        s = parse_console(check_for_unknown_keys, section_name='Processing', config={'Processing':self.processing})
        assert s == ''

        s = parse_console(check_for_unknown_keys, section_name='Curve Fit', config={'Curve Fit':self.curvefit})
        assert s == ''

        s = parse_console(check_for_unknown_keys, section_name='F Statistic', config={'F Statistic':self.fstatistic})
        assert s == ''

        s = parse_console(check_for_unknown_keys, section_name='Dashboard', config={'Dashboard':self.dashboard})
        assert s == ''


    def test_unknown_parameter_error(self):
        #
        # Meta
        #
        self.meta['unknown_key'] = None
        s = parse_console(check_for_unknown_keys, section_name='Meta', config={'Meta':self.meta})
        del self.meta['unknown_key']

        expected = "TOML Warning: In [Meta]: the key 'unknown_key' is unknown. Please double check !"
        expected = build_console_str(expected, ui.TerminalFormatting.WARNING)
        assert s == expected

        #
        # Experiment
        #
        self.experiment['unknown_key'] = None
        s = parse_console(check_for_unknown_keys, section_name='Experiment', config={'Experiment':self.experiment})
        del self.experiment['unknown_key']

        expected = "TOML Warning: In [Experiment]: the key 'unknown_key' is unknown. Please double check !"
        expected = build_console_str(expected, ui.TerminalFormatting.WARNING)
        assert s == expected

        #
        # Paths
        #
        self.paths['unknown_key'] = None
        s = parse_console(check_for_unknown_keys, section_name='Paths', config={'Paths':self.paths})
        del self.paths['unknown_key']

        expected = "TOML Warning: In [Paths]: the key 'unknown_key' is unknown. Please double check !"
        expected = build_console_str(expected, ui.TerminalFormatting.WARNING)
        assert s == expected

        #
        # Processing
        #
        self.processing['unknown_key'] = None
        s = parse_console(check_for_unknown_keys, section_name='Processing', config={'Processing':self.processing})
        del self.processing['unknown_key']

        expected = "TOML Warning: In [Processing]: the key 'unknown_key' is unknown. Please double check !"
        expected = build_console_str(expected, ui.TerminalFormatting.WARNING)
        assert s == expected

        #
        # Curve Fit
        #
        self.curvefit['unknown_key'] = None
        s = parse_console(check_for_unknown_keys, section_name='Curve Fit', config={'Curve Fit':self.curvefit})
        del self.curvefit['unknown_key']

        expected = "TOML Warning: In [Curve Fit]: the key 'unknown_key' is unknown. Please double check !"
        expected = build_console_str(expected, ui.TerminalFormatting.WARNING)
        assert s == expected

        #
        # F Statistic
        #
        self.fstatistic['unknown_key'] = None
        s = parse_console(check_for_unknown_keys, section_name='F Statistic', config={'F Statistic':self.fstatistic})
        del self.fstatistic['unknown_key']

        expected = "TOML Warning: In [F Statistic]: the key 'unknown_key' is unknown. Please double check !"
        expected = build_console_str(expected, ui.TerminalFormatting.WARNING)
        assert s == expected

        #
        # Dashboard
        #
        self.dashboard['unknown_key'] = None
        s = parse_console(check_for_unknown_keys, section_name='Dashboard', config={'Dashboard':self.dashboard})
        del self.dashboard['unknown_key']

        expected = "TOML Warning: In [Dashboard]: the key 'unknown_key' is unknown. Please double check !"
        expected = build_console_str(expected, ui.TerminalFormatting.WARNING)
        assert s == expected


class TestRequiredParameter(unittest.TestCase):
    test_config = {'Test Section': {'key X': None, 'key Y': None}}

    def test_no_required_parameter_no_error(self):
        key_list = []
        s = parse_console(check_for_required_keys, section_name='Test Section', config=TestRequiredParameter.test_config, key_list=key_list)
        assert s == ''

    def test_one_required_parameter_no_error(self):
        key_list = ['key X']
        s = parse_console(check_for_required_keys, section_name='Test Section', config=TestRequiredParameter.test_config, key_list=key_list)
        assert s == ''

    def test_two_required_parameter_no_error(self):
        key_list = ['key X', 'key Y']
        s = parse_console(check_for_required_keys, section_name='Test Section', config=TestRequiredParameter.test_config, key_list=key_list)
        assert s == ''

    def test_required_parameter_error(self):
        key_list = ['key Z']
        s = parse_console(check_for_required_keys, section_name='Test Section', config=TestRequiredParameter.test_config, key_list=key_list)
        expected = "TOML Error: In [Test Section]: the required key 'key Z' is missing. Please add."
        expected = build_console_str(expected, ui.TerminalFormatting.FAIL)
        expected = (ui.errorline() + '\n\n' + expected)
        assert s == expected


class TestCorrectValue(unittest.TestCase):
    test_config = {'Test Section': {'key X': 2, 'key Y': 'some text', 'key Z': False}}

    def test_empty_keylist_no_error(self):
        key_list = []
        requirement = (lambda v: type(v) is bool)
        s = parse_console(check_for_correct_values, section_name='Test Section', config=TestCorrectValue.test_config, key_list=key_list, requirement=requirement)
        assert s == ''

    def test_absent_key_no_error(self):
        key_list = ['key A']
        requirement = (lambda v: type(v) is bool)
        s = parse_console(check_for_correct_values, section_name='Test Section', config=TestCorrectValue.test_config, key_list=key_list, requirement=requirement)
        assert s == ''

    def test_different_requirements_no_error(self):
        key_list = ['key X']
        requirement = (lambda v: type(v) is int)
        s = parse_console(check_for_correct_values, section_name='Test Section', config=TestCorrectValue.test_config, key_list=key_list, requirement=requirement)
        assert s == ''

        key_list = ['key Y']
        requirement = (lambda v: type(v) is str)
        s = parse_console(check_for_correct_values, section_name='Test Section', config=TestCorrectValue.test_config, key_list=key_list, requirement=requirement)
        assert s == ''

        key_list = ['key Z']
        requirement = (lambda v: type(v) is bool)
        s = parse_console(check_for_correct_values, section_name='Test Section', config=TestCorrectValue.test_config, key_list=key_list, requirement=requirement)
        assert s == ''

    def test_wrong_requirement_error(self):
        key_list = ['key Y']
        requirement = (lambda v: type(v) is int)
        s = parse_console(check_for_correct_values, section_name='Test Section', config=TestCorrectValue.test_config, key_list=key_list, requirement=requirement)
        expected = "TOML Error: In [Test Section]: the key-value pair 'key Y = some text' does not meet the specifications. Please fix."
        expected = build_console_str(expected, ui.TerminalFormatting.FAIL)
        expected = (ui.errorline() + '\n\n' + expected)
        assert s == expected


class TestExperimentSetUp():
    test_config = {
        'Experiment': {
            'experiments': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'control_experiment': np.array([1, 2]),
            'doses': np.array([0., 0., 1., 1., 10., 10., 100., 100., 1000., 1000.]),
            'dose_scale': '1e-6',
            'dose_unit': 'M'
    }}

    def test_no_error(self):
        test_config = copy.deepcopy(TestExperimentSetUp.test_config)
        s = parse_console(check_correct_experiment_setup, config=test_config)
        assert s == ''

    def test_too_little_experiments_error(self):
        test_config = copy.deepcopy(TestExperimentSetUp.test_config)
        test_config['Experiment']['experiments'] = np.array([])

        s = parse_console(check_correct_experiment_setup, config=test_config)
        expected = "TOML Error: [Experiment] 'experiments' and [Experiment] 'doses' need at least length 4."
        expected = build_console_str(expected, ui.TerminalFormatting.FAIL)
        expected = (ui.errorline() + '\n\n' + expected)
        assert s == expected

    def test_too_little_doses_error(self):
        test_config = copy.deepcopy(TestExperimentSetUp.test_config)
        test_config['Experiment']['doses'] = np.array([])

        s = parse_console(check_correct_experiment_setup, config=test_config)
        expected = "TOML Error: [Experiment] 'experiments' and [Experiment] 'doses' need at least length 4."
        expected = build_console_str(expected, ui.TerminalFormatting.FAIL)
        expected = (ui.errorline() + '\n\n' + expected)
        assert s == expected

    def test_different_legnth_error(self):
        test_config = copy.deepcopy(TestExperimentSetUp.test_config)
        test_config['Experiment']['experiments'] = np.array([1,2,3,4,5,6])

        s = parse_console(check_correct_experiment_setup, config=test_config)
        expected = "TOML Error: [Experiment] 'experiments' and [Experiment] 'doses' do no correspond in length."
        expected = build_console_str(expected, ui.TerminalFormatting.FAIL)
        expected = (ui.errorline() + '\n\n' + expected)
        assert s == expected

        test_config = copy.deepcopy(TestExperimentSetUp.test_config)
        test_config['Experiment']['doses'] = np.array([0., 0., 1., 1., 10., 10.])

        s = parse_console(check_correct_experiment_setup, config=test_config)
        expected = "TOML Error: [Experiment] 'experiments' and [Experiment] 'doses' do no correspond in length."
        expected = build_console_str(expected, ui.TerminalFormatting.FAIL)
        expected = (ui.errorline() + '\n\n' + expected)
        assert s == expected

    def test_control_not_in_experiments(self):
        test_config = copy.deepcopy(TestExperimentSetUp.test_config)
        test_config['Experiment']['control_experiment'] = np.array([-1, 1, 2])

        s = parse_console(check_correct_experiment_setup, config=test_config)
        expected = "TOML Error: [Experiment] at least one 'control_experiment' is not in [Experiment] 'experiments'."
        expected = build_console_str(expected, ui.TerminalFormatting.FAIL)
        expected = (ui.errorline() + '\n\n' + expected)
        assert s == expected

    def test_dose_scale(self):
        test_config = copy.deepcopy(TestExperimentSetUp.test_config)
        test_config['Experiment']['dose_scale'] = ''

        s = parse_console(check_correct_experiment_setup, config=test_config)
        expected = "TOML Error: [Experiment] 'dose_scale' is empty."
        expected = build_console_str(expected, ui.TerminalFormatting.FAIL)
        expected = (ui.errorline() + '\n\n' + expected)
        assert s == expected

        test_config['Experiment']['dose_scale'] = None

        s = parse_console(check_correct_experiment_setup, config=test_config)
        expected = "TOML Error: [Experiment] 'dose_scale' is empty."
        expected = build_console_str(expected, ui.TerminalFormatting.FAIL)
        expected = (ui.errorline() + '\n\n' + expected)
        assert s == expected

    def test_dose_unit(self):
        test_config = copy.deepcopy(TestExperimentSetUp.test_config)
        test_config['Experiment']['dose_unit'] = ''

        s = parse_console(check_correct_experiment_setup, config=test_config)
        expected = "TOML Error: [Experiment] 'dose_unit' is empty."
        expected = build_console_str(expected, ui.TerminalFormatting.FAIL)
        expected = (ui.errorline() + '\n\n' + expected)
        assert s == expected

    def test_unique_experiments(self):
        test_config = copy.deepcopy(TestExperimentSetUp.test_config)
        test_config['Experiment']['experiments'] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 9])

        s = parse_console(check_correct_experiment_setup, config=test_config)
        expected = "TOML Error: [Experiment] 'experiments' contains duplicates. Make sure experiment names are unique."
        expected = build_console_str(expected, ui.TerminalFormatting.FAIL)
        expected = (ui.errorline() + '\n\n' + expected)
        assert s == expected

    def test_unique_control_experiments(self):
        test_config = copy.deepcopy(TestExperimentSetUp.test_config)
        test_config['Experiment']['control_experiment'] = np.array([1, 1])

        s = parse_console(check_correct_experiment_setup, config=test_config)
        expected =  "TOML Error: [Experiment] 'control_experiment' contains duplicates. Make sure experiment names are unique."
        expected = build_console_str(expected, ui.TerminalFormatting.FAIL)
        expected = (ui.errorline() + '\n\n' + expected)
        assert s == expected
