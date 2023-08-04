import numpy as np
import pandas as pd
import re


class MaxQuantMap:
    def __init__(self, version):
        version = version.split('.')
        self.version_major = version[0]
        self.version_minor = version[1]

    @staticmethod
    def rename_general_columns(cols):
        return cols.map(lambda c: MaxQuantMap._col_map.get(c, c))

    @staticmethod
    def map_indicator_values(df):
        for col in MaxQuantMap._indicator_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda v: MaxQuantMap._indicator_map.get(v, False))
        return df

    @staticmethod
    def rename_tmt_columns(cols):
        return cols.str.replace(r'Reporter intensity corrected', 'Raw', regex=True)

    @staticmethod
    def rename_lfq_columns(cols):
        return cols.str.replace(r'LFQ intensity', 'Raw', regex=True)

    @staticmethod
    def restructure_lfq_evidence(df):
        index = ['Modified sequence', 'Proteins', 'Genes', 'Charge', 'Contaminant', 'Decoy']
        df_new = pd.pivot_table(data=df, aggfunc=np.sum, values='Intensity', index=index, columns='Experiment')
        df_new = df_new.rename(columns=lambda i: f'Intensity {i}')
        df_new['Score'] = df.groupby(index)['Score'].max()
        df_new.reset_index(inplace=True)
        return df_new

    @staticmethod
    def rename_intensity_columns(cols):
        return cols.str.replace(r'Intensity', 'Raw', regex=True)

    @staticmethod
    def clean_protein_group_tmt_cols(col):
        patterns = [
            r'(?P<col>Reporter intensity \d+).*',
            r'(?P<col>Reporter intensity corrected \d+).*',
            r'(?P<col>Reporter intensity count \d+).*',
        ]
        for p in patterns:
            match = re.fullmatch(p, col)
            if match:
                return match['col']
        return col

    _col_map = {
        'Modified Sequence': 'Modified sequence',
        'Potential contaminant': 'Contaminant',
        'Reverse': 'Decoy',
        'Gene names': 'Genes',
        'Protein IDs': 'Proteins',
    }

    _indicator_cols = ['Contaminant', 'Decoy']

    _indicator_map = {
        '+': True,
        np.nan: False,
        '': False,
    }
