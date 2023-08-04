import numpy as np
import curve_curator.toolbox as toolbox


class TestBuildDrugLogConcentrations:
    concentrations = np.array([0.0, 0.1, 1, 10, 100])
    scale = 1e-6
    dmso_offset = 1e3

    def test_empty_input(self):
        concentrations = []
        output = toolbox.build_drug_log_concentrations(concentrations, scale=self.scale, dmso_offset=self.dmso_offset)
        assert output.shape == (0,)

    def test_single_value(self):
        concentrations = np.array([1.0])
        expected_output = np.array([-6.0])
        output = toolbox.build_drug_log_concentrations(concentrations, scale=self.scale, dmso_offset=self.dmso_offset)
        np.testing.assert_almost_equal(output, expected_output, decimal=3)

    def test_case_1(self):
        expected_output = np.array([-10.0, -7.0, -6.0, -5.0, -4.0])
        output = toolbox.build_drug_log_concentrations(self.concentrations, scale=self.scale, dmso_offset=self.dmso_offset)
        np.testing.assert_almost_equal(output, expected_output, decimal=3)

    def test_case_2(self):
        scale = 1e-9
        expected_output = np.array([-13.0, -10.0, -9.0, -8.0, -7.0])
        output = toolbox.build_drug_log_concentrations(self.concentrations, scale=scale, dmso_offset=self.dmso_offset)
        np.testing.assert_almost_equal(output, expected_output, decimal=3)

    def test_case_3(self):
        dmso_offset = 1e6
        expected_output = np.array([-13.0, -7.0, -6.0, -5.0, -4.0])
        output = toolbox.build_drug_log_concentrations(self.concentrations, scale=self.scale, dmso_offset=dmso_offset)
        np.testing.assert_almost_equal(output, expected_output, decimal=3)
