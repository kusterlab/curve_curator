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


class TestXYAggregation:

    def test_empty_input(self):
        x, y = [], []
        x_expected, y_expected = np.array([]), np.array([])
        x_agg, y_agg = toolbox.aggregate_xy(x, y)
        np.testing.assert_almost_equal(x_agg, x_expected)
        np.testing.assert_almost_equal(y_agg, y_expected)

    def test_single_input(self):
        x, y = [1], [5]
        x_expected, y_expected = np.array([1]), np.array([5])
        x_agg, y_agg = toolbox.aggregate_xy(x, y)
        np.testing.assert_almost_equal(x_agg, x_expected)
        np.testing.assert_almost_equal(y_agg, y_expected)

    def test_replicated_input(self):
        x, y = [1, 1, 1], [4, 5, 6]
        x_expected, y_expected = np.array([1]), np.array([5])
        x_agg, y_agg = toolbox.aggregate_xy(x, y)
        np.testing.assert_almost_equal(x_agg, x_expected)
        np.testing.assert_almost_equal(y_agg, y_expected)

    def test_float_int_mix(self):
        x, y = [1, 1.0, 1], [4, 5, 6]
        x_expected, y_expected = np.array([1]), np.array([5])
        x_agg, y_agg = toolbox.aggregate_xy(x, y)
        np.testing.assert_almost_equal(x_agg, x_expected)
        np.testing.assert_almost_equal(y_agg, y_expected)

    def test_real_example(self):
        x = np.concatenate(3 * [np.arange(-10, -4, 1.0)])
        x_expected = np.arange(-10, -4, 1.0)
        y = np.concatenate([np.arange(1, 7, 1.0), np.arange(2, 8, 1.0), np.arange(3, 9, 1.0)])
        y_expected = np.arange(2, 8, 1.0)
        x_agg, y_agg = toolbox.aggregate_xy(x, y)
        np.testing.assert_almost_equal(x_agg, x_expected)
        np.testing.assert_almost_equal(y_agg, y_expected)

    def test_missing_value_presence(self):
        x = np.concatenate(3 * [np.arange(-10, -4, 1.0)])
        x_expected = np.arange(-10, -4, 1.0)
        y = np.concatenate([np.arange(1, 7, 1.0), np.arange(2, 8, 1.0), np.arange(3, 9, 1.0)])
        y_expected = np.arange(2, 8, 1.0)

        y[8] = np.nan
        x_agg, y_agg = toolbox.aggregate_xy(x, y)
        np.testing.assert_almost_equal(x_agg, x_expected)
        np.testing.assert_almost_equal(y_agg, y_expected)

        y[[2, 8, 14]] = np.nan
        y_expected[2] = np.nan
        x_agg, y_agg = toolbox.aggregate_xy(x,y)
        np.testing.assert_almost_equal(x_agg, x_expected)
        np.testing.assert_almost_equal(y_agg, y_expected)


class TestRoundingUp:

    def test_zero(self):
        x = 0.0
        expected = 0.0
        assert expected == toolbox.roundup(x)

    def test_one(self):
        x = 1.0
        expected = 1.0
        assert expected == toolbox.roundup(x)

    def test_one_minus(self):
        x = -1.0
        expected = -1.0
        assert expected == toolbox.roundup(x)

    def test_1decimal(self):
        x = 1.1
        expected = 2.0
        assert expected == toolbox.roundup(x)
        x = 0.9
        expected = 0.9
        assert expected == toolbox.roundup(x)

    def test_1decimal_minus(self):
        x = -1.1
        expected = -1.0
        assert expected == toolbox.roundup(x)
        x = -0.9
        expected = -0.1
        assert expected == toolbox.roundup(x)

    def test_3decimal(self):
        x = 1.111
        expected = 2.0
        assert expected == toolbox.roundup(x)

    def test_twenty(self):
        x = 11.1
        expected = 20.0
        assert expected == toolbox.roundup(x)


class TestRoundingDown:

    def test_zero(self):
        x = 0.0
        expected = 0.0
        assert expected == toolbox.rounddown(x)

    def test_one(self):
        x = 1.0
        expected = 1.0
        assert expected == toolbox.rounddown(x)

    def test_one_minus(self):
        x = -1.0
        expected = -1.0
        assert expected == toolbox.roundup(x)

    def test_1decimal(self):
        x = 0.9
        expected = 0.1
        assert expected == toolbox.rounddown(x)
        x = 1.1
        expected = 1.0
        assert expected == toolbox.rounddown(x)

    def test_1decimal_minus(self):
        x = -0.9
        expected = -0.9
        assert expected == toolbox.rounddown(x)
        x = -1.1
        expected = -2.0
        assert expected == toolbox.rounddown(x)

    def test_3decimal(self):
        x = 0.999
        expected = 0.1
        assert expected == toolbox.rounddown(x)

    def test_twenty(self):
        x = 11.1
        expected = 10.0
        assert expected == toolbox.rounddown(x)
