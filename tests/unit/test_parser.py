import pandas as pd
import numpy as np
from curve_curator.data_parser import clean_modified_sequence, clean_rows, aggregate_duplicates


class TestCleanModifiedSequence:
    def test_empty_imput(self):
        mod_seq = pd.Series([], dtype=str)
        expected_result = pd.Series([], dtype=str)
        assert clean_modified_sequence(mod_seq).equals(expected_result)

    def test_no_modifications(self):
        # Test case 1: No replacements
        mod_seq = pd.Series(['_ABC_', '_DEF_', 'ABC', 'DEF'])
        expected_result = pd.Series(['ABC', 'DEF', 'ABC', 'DEF'])
        assert clean_modified_sequence(mod_seq).equals(expected_result)

    def test_modifications(self):
        mod_seq = pd.Series(['_(Acetyl (Protein N-term))ABCXYZ_', '_DEFpSpTpY_', '_GHIM(Oxidation (M))K_', '_LNPK(GG (K))R_'])
        expected_result = pd.Series(['(ac)ABCXYZ', 'DEFS(ph)T(ph)Y(ph)', 'GHIMK', 'LNPK(ub)R'])
        assert clean_modified_sequence(mod_seq).equals(expected_result)

    def test_multiple_modifications(self):
        mod_seq = pd.Series(['_(Acetyl (Protein N-term))ABCpSpTpY_', '_DEFM(ox)pSpT_'])
        expected_result = pd.Series(['(ac)ABCS(ph)T(ph)Y(ph)', 'DEFMS(ph)T(ph)'])
        assert clean_modified_sequence(mod_seq).equals(expected_result)


class TestCleanRows:
    def test_empty_imput(self):
        df = pd.DataFrame({})
        expected_result = pd.DataFrame({})
        assert clean_rows(df).equals(expected_result)

    def test_contaminant_filtering(self):
        df = pd.DataFrame({
            'ID': [1, 2, 3, 4],
            'Sequence': ['ABC', 'DEF', 'GHI', 'JKL'],
            'Contaminant': [False, True, False, False],
        })
        expected_result = pd.DataFrame({
            'ID': [1, 3, 4],
            'Sequence': ['ABC', 'GHI', 'JKL'],
            'Contaminant': [False, False, False],
        })
        assert clean_rows(df).equals(expected_result)

    def test_decoy_filtering(self):
        df = pd.DataFrame({
            'ID': [1, 2, 3, 4],
            'Sequence': ['ABC', 'DEF', 'GHI', 'JKL'],
            'Decoy': [False, False, True, False],
        })
        expected_result = pd.DataFrame({
            'ID': [1, 2, 4],
            'Sequence': ['ABC', 'DEF', 'JKL'],
            'Decoy': [False, False, False],
        })
        assert clean_rows(df).equals(expected_result)

    def test_decoy_and_contaminant_filtering(self):
        df = pd.DataFrame({
            'ID': [1, 2, 3, 4],
            'Sequence': ['ABC', 'DEF', 'GHI', 'JKL'],
            'Contaminant': [False, True, False, False],
            'Decoy': [False, False, True, False],
        })
        expected_result = pd.DataFrame({
            'ID': [1, 4],
            'Sequence': ['ABC', 'JKL'],
            'Contaminant': [False, False],
            'Decoy': [False, False],
        })
        assert clean_rows(df).equals(expected_result)

    def test_decoy_and_contaminant_absent(self):
        df = pd.DataFrame({
            'ID': [1, 2, 3, 4],
            'Sequence': ['ABC', 'DEF', 'GHI', 'JKL']
        })
        expected_result = pd.DataFrame({
            'ID': [1, 2, 3, 4],
            'Sequence': ['ABC', 'DEF', 'GHI', 'JKL']
        })
        assert clean_rows(df).equals(expected_result)


class TestAggregateDuplicates:
    df = pd.DataFrame({
        'ID': [1, 1, 2, 2, 3, 4, 5, 5],
        'Value1': [10, 20, 30, 40, 50, 0, np.nan, np.nan],
        'Value2': [1, 2, 3, 4, 5, np.nan, 6, np.nan],
        'Category': ['A', 'B', 'C', 'C', 'A', 'D', 'D', 'D'],
        'Name': ['A', 'A', 'B', 'B', 'C', 'D', 'E', 'E'],
        'Boolean': [True, True, False, False, False, False, False, False],
    })
    keys = ['ID']
    numeric_cols = ['Value1', 'Value2']
    feature_cols = ['Name', 'Boolean']
    category_cols = ['Category', 'Name', 'Boolean']

    def test_key_only(self):
        result = aggregate_duplicates(self.df, self.keys)
        expected_result = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],
            'N duplicates': [2, 2, 1, 1, 2],
        })
        assert result.equals(expected_result)

    def test_sumcols(self):
        result = aggregate_duplicates(self.df, self.keys, sum_cols=self.numeric_cols)
        expected_result = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],
            'N duplicates': [2, 2, 1, 1, 2],
            'Value1': [30, 70, 50, 0, np.nan],
            'Value2': [3, 7, 5, np.nan, 6],
        })
        assert result.equals(expected_result)

    def test_maxcols(self):
        result = aggregate_duplicates(self.df, self.keys, max_cols=self.numeric_cols)
        expected_result = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],
            'N duplicates': [2, 2, 1, 1, 2],
            'Value1': [20, 40, 50, 0, np.nan],
            'Value2': [2, 4, 5, np.nan, 6],
        })
        assert result.equals(expected_result)

    def test_mincols(self):
        result = aggregate_duplicates(self.df, self.keys, min_cols=self.numeric_cols)
        expected_result = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],
            'N duplicates': [2, 2, 1, 1, 2],
            'Value1': [10, 30, 50, 0, np.nan],
            'Value2': [1, 3, 5, np.nan, 6],
        })
        assert result.equals(expected_result)

    def test_firstcols(self):
        result = aggregate_duplicates(self.df, self.keys, first_cols=self.feature_cols)
        expected_result = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],
            'N duplicates': [2, 2, 1, 1, 2],
            'Name': ['A', 'B', 'C', 'D', 'E'],
            'Boolean': [True, False, False, False, False],
        })
        assert result.equals(expected_result)

    def test_concatcols(self):
        result = aggregate_duplicates(self.df, self.keys, concat_cols=self.category_cols)
        expected_result = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],
            'N duplicates': [2, 2, 1, 1, 2],
            'Category': ['A;B', 'C;C', 'A', 'D', 'D;D'],
            'Name': ['A;A', 'B;B', 'C', 'D', 'E;E'],
            'Boolean': ['True;True', 'False;False', 'False', 'False', 'False;False'],
        })
        assert result.equals(expected_result)
