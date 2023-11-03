import numpy as np
import re


class PDMap:
    def __init__(self, version):
        version = version.split('.')
        self.version_major = version[0]
        self.version_minor = version[1]

    @staticmethod
    def rename_general_columns(cols):
        return cols.map(lambda c: PDMap._col_map.get(c, c))

    @staticmethod
    def map_indicator_values(df):
        for col in PDMap._indicator_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda v: PDMap._indicator_map.get(v, False))
        return df

    @staticmethod
    def rename_tmt_columns(cols):
        cols = cols.str.replace(r'Abundance:', 'Raw', regex=True)
        for i, c in enumerate(PDMap._tmt_channel_names):
            cols = cols.str.replace(c, str(i + 1), regex=True)
        return cols

    @staticmethod
    def rename_intensity_columns(cols):
        return cols.str.replace(r'Abundances Grouped F', 'Raw ', regex=True)

    @staticmethod
    def create_mod_sequence(df):
        # Remove flanking sequences if present
        flanking_mask = df['Modified sequence'].str.match(r'\[.+\]\..+.\.\[.+\]')
        df.loc[flanking_mask, 'Modified sequence'] = df.loc[flanking_mask, 'Modified sequence'].str.split('.', expand=True)[1]
        df['Modified sequence'] = df['Modified sequence'].str.upper()

        # transform known modifications and report unknown ones to the user
        unknown_mods = set()
        df['Modifications dict'] = df['Modifications'].apply(to_pos_dict, unknown_mods=unknown_mods)
        df['Modified sequence'] = df[['Modified sequence', 'Modifications dict']].apply(to_mod_seq, axis=1)
        df.drop(columns=['Modifications dict'], inplace=True)
        if unknown_mods:
            print(' * The following unknown modifications were detected: ', unknown_mods, end='\n\n')
        return df

    _col_map = {
        'Accession': 'Proteins',
        'Protein Accessions': 'Proteins',
        'Annotated Sequence': 'Modified sequence',
        'Contaminant': 'Contaminant',
        'Gene ID': 'Genes',
        'Number of Peptides': 'Peptides',
        'Sum PEP Score': 'Score',
    }

    _mod_map = {
        'Phospho': 'ph',
        'Acetyl': 'ac',
        'Oxidation': 'ox',
        'Carbamidomethyl': '',
    }

    _tmt_channel_names = ['126', '127N', '127C', '128N', '128C', '129N', '129C', '130N', '130C', '131N', '131C', '132N', '132C', '133N', '133C',
                          '134N', '134C', '135N']

    _indicator_cols = ['Contaminant', 'Decoy']

    _indicator_map = {
        '+': True,
        np.nan: False,
        '': False,
    }


def to_pos_dict(s, mod_pattern=r'(?P<pos>.*)\((?P<name>.*)\)', pos_pattern=r'.(?P<pos>\d+)', unknown_mods=set()):
    """
    Parses the modification string into a dictionary (<pos>:'mod_name", ...)
    """
    d = {}
    try:
        for mod in str(s).split(';'):
            if mod:
                mod_match = re.fullmatch(mod_pattern, mod.strip())

                # Handle N-terminal modifications
                if 'N-Term(Prot)' == mod_match['pos']:
                    d[0] = PDMap._mod_map.get(mod_match['name'], mod_match['name'])
                    continue

                # Ignore TMT modifications
                if 'TMT' not in mod_match['name']:
                    pos_match = re.fullmatch(pos_pattern, mod_match['pos'])
                    d[int(pos_match['pos'])] = PDMap._mod_map.get(mod_match['name'], str(mod_match['name']).lower())

                    if mod_match['name'] not in PDMap._mod_map:
                        unknown_mods.add(mod_match['name'])
            else:
                print(s)
    except TypeError:
        print(mod_match['pos'], mod_match['name'])
    return d


def to_mod_seq(row):
    """
    creates a modified sequence string based on the input
    """
    seq, mod_dict = row
    mod_seq = []
    n = len(seq)
    i = 0
    for p, mod in sorted(mod_dict.items()):
        mod_seq.append(seq[i:p])
        mod_seq.append(f'({mod})')
        i = p
    mod_seq.append(seq[i:n])
    return ''.join(mod_seq)