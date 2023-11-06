import numpy as np


class FraggerMap:
    def __init__(self, version):
        version = version.split('.')
        self.version_major = version[0]
        self.version_minor = version[1]

    @staticmethod
    def rename_general_columns(cols):
        return cols.map(lambda c: FraggerMap._col_map.get(c, c))

    @staticmethod
    def rename_tmt_columns(cols):
        return cols.str.replace(r'_', ' ', regex=True)

    @staticmethod
    def rename_lfq_columns(cols):
        cols = cols.str.replace(r' Intensity', '', regex=True)
        cols = cols.str.replace(r'_', ' ', regex=True)
        return cols

    @staticmethod
    def create_mod_sequence(df):
        # Update registered modifications
        for old, new in FraggerMap._mod_map.items():
            df['Modified sequence'] = df['Modified sequence'].str.replace(old, new)
        # report unknown mods if present
        unknown_mods = set.union(*df['Modified sequence'].str.extractall(r'(.\[\d+])')[0].unstack(level='match').apply(set, axis=1), {np.nan}) - {np.nan}
        if unknown_mods:
            print(' * The following unknown modifications were detected: ', unknown_mods, end='\n\n')
        return df

    _col_map = {
        'Protein ID': 'Proteins',
        'Modified Sequence': 'Modified sequence',
        'Gene': 'Genes',
        'Spectral Count': 'PSMs',
        'Sum PEP Score': 'Score',
        'Total Peptides': 'Peptides',
        'Combined Total Peptides': 'Peptides',
    }

    _mod_map = {
        # There is a difference between LFQ and TMT search for MS fragger how they annotate mod masses.
        # The exact Mod mass is LFQ. The total rounded residue is TMT.
        'n[230]': '',   # TMT n-terminus
        'n[42.0106]': '',
        'K[357]': 'K',  # TMT lysine
        'S[167]': 'S(ph)',
        'S[79.9663]': 'S(ph)',
        'T[181]': 'T(ph)',
        'T[79.9663]': 'T(ph)',
        'Y[243]': 'Y(ph)',
        'Y[79.9663]': 'Y(ph)',
        'K[170]': 'K(ac)',
        'K[42.0106]': 'K(ac)',
        'M[147]': 'M(ox)',
        'M[15.9949]': 'M(ox)',
        'C[160]': 'C',   # Carbamidomethyl
        'C[57.0215]': 'C',  # Carbamidomethyl
    }
