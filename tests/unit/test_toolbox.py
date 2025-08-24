import time
import numpy as np
import pandas as pd
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


class TestColNameGenerator:

    def test_with_numbers(self):
        expected = ['Test 0', 'Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5']
        out = toolbox.build_col_names('Test {}', range(6))
        assert all(out == expected)

    def test_with_strings(self):
        expected = ['Test A', 'Test B', 'Test C', 'Test D']
        out = toolbox.build_col_names('Test {}', [*'ABCD'])
        assert all(out == expected)


class TestParallelization:

    df = pd.DataFrame({
        'A': np.random.normal(size=100),
        'B': np.random.normal(size=100),
        'C': np.random.normal(size=100),
    })
    df.index.name='Original index'
    return_cols = ['Mean Multi', 'STD Multi']
    func_kwargs = {'c':1, 'd':2,}

    @staticmethod
    def slow_func(row, c=0, d=0, **kwargs):
        time.sleep(np.random.uniform(0, 0.05))
        out = np.mean(row) + c, np.std(row, ddof=1) + d  # np and pd have different base definitions!
        return (row.name, *out)

    def calculate_expected_output(self, c=0, d=0):
        out = pd.concat([
            self.df.mean(axis=1).rename(self.return_cols[0]) + c,
            self.df.std(axis=1).rename(self.return_cols[1]) + d],
            axis=1)
        return out


    #
    # Test one core. The index is preserved with one core
    def test_one_core(self):
        in_df = self.df.copy()
        expected_df = self.calculate_expected_output()
        out_df = toolbox.parallelize_dataframe(
            df=in_df,
            func=self.slow_func,
            return_cols=self.return_cols,
            n_cores=1,
            sorted=False,
        )
        assert all(in_df.index.values == out_df.index.values) and all(expected_df.index.values == out_df.index.values)

        out_df = toolbox.parallelize_dataframe(
            df=in_df,
            func=self.slow_func,
            return_cols=self.return_cols,
            n_cores=1,
            sorted=True,
        )
        assert all(in_df.index.values == out_df.index.values) and all(expected_df.index.values == out_df.index.values)
        print(pd.merge(expected_df[self.return_cols[0]], out_df[self.return_cols[0]], left_index=True, right_index=True))
        pd.testing.assert_frame_equal(out_df, expected_df)

    #
    # Test multiple cores. The index order is not preserved unless sorted.
    def test_multiple_cores(self):
        in_df = self.df.copy()
        expected_df = self.calculate_expected_output()

        out_df = toolbox.parallelize_dataframe(
            df=in_df,
            func=self.slow_func,
            return_cols=self.return_cols,
            n_cores=4,
            sorted=False,
        )
        assert (not all(in_df.index.values == out_df.index.values)) or (not all(expected_df.index.values == out_df.index.values))

        out_df = toolbox.parallelize_dataframe(
            df=in_df,
            func=self.slow_func,
            return_cols=self.return_cols,
            n_cores=4,
            sorted=True,
        )
        assert all(in_df.index.values == out_df.index.values) and all(expected_df.index.values == out_df.index.values)
        pd.testing.assert_frame_equal(out_df, expected_df)

    #
    # Test passing of kwargs to parallelized func
    def test_kwargs(self):
        in_df = self.df.copy()
        expected_df = self.calculate_expected_output(**self.func_kwargs)
        out_df = toolbox.parallelize_dataframe(
            df=in_df,
            func=self.slow_func,
            func_kwargs=self.func_kwargs,
            return_cols=self.return_cols,
            n_cores=4,
            sorted=True,
        )
        pd.testing.assert_frame_equal(out_df, expected_df)
