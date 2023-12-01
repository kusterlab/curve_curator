import numpy as np
import pandas as pd
import curve_curator.quantification as quantification


class TestFilterNans:
    df = pd.DataFrame({
        'A': [1.0, 2.0, np.nan, 4.0],
        'B': [np.nan, 6.0, np.nan, 8.0],
        'C': [9.0, 10.0, np.nan, np.nan],
        'D': [13.0, 14.0, 15.0, 16.0],
    })

    def test_filter_nans_no_missing(self):
        result = quantification.filter_nans(self.df.copy(deep=True), ['A', 'B', 'C', 'D'], 0)
        expected = pd.DataFrame({'A': [2.0], 'B': [6.0], 'C': [10.0], 'D': [14.0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_filter_nans_some_missing(self):
        result = quantification.filter_nans(self.df.copy(deep=True), ['A', 'B', 'C', 'D'], 1)
        expected = pd.DataFrame({'A': [1.0, 2.0, 4.0], 'B': [np.nan, 6.0, 8.0], 'C': [9.0, 10.0, np.nan], 'D': [13.0, 14.0, 16.0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_filter_nans_no_missing_on_subset_columns(self):
        result = quantification.filter_nans(self.df.copy(deep=True), ['A', 'B', 'D'], 0)
        expected = pd.DataFrame({'A': [2.0, 4.0], 'B': [6.0, 8.0], 'C': [10.0, np.nan], 'D': [14.0, 16.0]})
        pd.testing.assert_frame_equal(result, expected)


class TestImputeNans:
    df = pd.DataFrame({
        'A': [1.0, 2.0, np.nan, 4.0],
        'B': [np.nan, 6.0, np.nan, 8.0],
        'C': [9.0, 10.0, np.nan, np.nan],
        'D': [13.0, 14.0, 15.0, 16.0],
    })
    imputation_value = 0.0
    max_imputations = np.inf

    def test_imputations_to0_nofiltering(self):
        result = quantification.impute_nans(self.df.copy(deep=True), ['A', 'B', 'C', 'D'], self.imputation_value, self.max_imputations)
        expected = pd.DataFrame({
            'A': [1.0, 2.0, 0.0, 4.0],
            'B': [0.0, 6.0, 0.0, 8.0],
            'C': [9.0, 10.0, 0.0, 0.0],
            'D': [13.0, 14.0, 15.0, 16.0],
            'Imputation N': [1, 0, 3, 1],
            'Imputation Position': ['B', '', 'A;B;C', 'C'],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_imputations_to2_nofiltering(self):
        result = quantification.impute_nans(self.df.copy(deep=True), ['A', 'B', 'C', 'D'], 2.0, self.max_imputations)
        expected = pd.DataFrame({
            'A': [2.0, 2.0, 2.0, 4.0],
            'B': [2.0, 6.0, 2.0, 8.0],
            'C': [9.0, 10.0, 2.0, 2.0],
            'D': [13.0, 14.0, 15.0, 16.0],
            'Imputation N': [2, 0, 3, 1],
            'Imputation Position': ['A;B', '', 'A;B;C', 'C'],
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_imputations_to1_filteringfor2(self):
        result = quantification.impute_nans(self.df.copy(deep=True), ['A', 'B', 'C', 'D'], self.imputation_value, 2)
        expected = pd.DataFrame({
            'A': [1.0, 2.0, 4.0],
            'B': [0.0, 6.0, 8.0],
            'C': [9.0, 10.0, 0.0],
            'D': [13.0, 14.0, 16.0],
            'Imputation N': [1, 0, 1],
            'Imputation Position': ['B', '', 'C'],
        })
        pd.testing.assert_frame_equal(result, expected)


class TestNormalizeValues:
    df = pd.DataFrame({
        'A': [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0],
        'B': [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 5.0],
        'C': [0.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0],
        'ref': [True, False, False, False, False, False, False, False, False, False, True]
    })
    ref_col = 'ref'
    raw_cols = ['A', 'B', 'C']
    norm_cols = ['a', 'b', 'c']

    def test_normalization_no_reference(self):
        result_df, result_factors = quantification.normalize_values(self.df.copy(deep=True), self.raw_cols, self.norm_cols)
        expected_df = pd.DataFrame({
            'a': [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 10.0],
            'b': [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 5.0],
            'c': [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.5],
        })
        expected_factors = pd.Series({'A': 1.0, 'B': 0.0, 'C': -1.0})
        pd.testing.assert_frame_equal(result_df[self.norm_cols], expected_df)
        pd.testing.assert_series_equal(result_factors[self.raw_cols], expected_factors)

    def test_normalization_with_reference(self):
        result_df, result_factors = quantification.normalize_values(self.df.copy(deep=True), self.raw_cols, self.norm_cols, self.ref_col)
        expected_df = pd.DataFrame({
            'a': [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0],
            'b': [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 5.0],
            'c': [0.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0],
        })
        print(result_df)
        expected_factors = pd.Series({'A': 0.0, 'B': 0.0, 'C': 0.0})
        pd.testing.assert_frame_equal(result_df[self.norm_cols], expected_df)
        pd.testing.assert_series_equal(result_factors[self.raw_cols], expected_factors)


class TestCalculateRatios:
    df = pd.DataFrame({
        'A': [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, np.nan],
        'B': [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 1.0, np.nan, 2.0],
        'C': [0.0, 2.0, 1.0, 3.0, 2.0, 4.0, 0.0, np.nan, 4.0],
    })
    raw_cols = ['A', 'B', 'C']
    ratio_cols = ['a', 'b', 'c']

    def test_ratios_with_single_control(self):
        ref_col = ['A']
        result_df = quantification.add_ratios(self.df.copy(deep=True), self.raw_cols, self.ratio_cols, ref_col)
        expected_df = pd.DataFrame({
            'a': [np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0,    1.0, np.nan],
            'b': [np.nan, np.nan, 1.0, 2.0, 1.0, 2.0, 0.5, np.nan, np.nan],
            'c': [np.nan, np.nan, 1.0, 3.0, 1.0, 2.0, 0.0, np.nan, np.nan],
        })
        pd.testing.assert_frame_equal(result_df[self.ratio_cols], expected_df)

    def test_ratios_with_multiple_controls(self):
        ref_col = ['A', 'B']
        result_df = quantification.add_ratios(self.df.copy(deep=True), self.raw_cols, self.ratio_cols, ref_col)
        expected_df = pd.DataFrame({
            'a': [np.nan, 0.0, 1.0, 2/3, 1.0, 2/3, 4/3, 1.0, np.nan],
            'b': [np.nan, 2.0, 1.0, 4/3, 1.0, 4/3, 2/3, np.nan, 1.0],
            'c': [np.nan, 4.0, 1.0, 2.0, 1.0, 4/3, 0.0, np.nan, 2.0],
        })
        pd.testing.assert_frame_equal(result_df[self.ratio_cols], expected_df)


class TestBuildInterpolationPoints:
    pass


class TestBuildDDWeights:
    pass
