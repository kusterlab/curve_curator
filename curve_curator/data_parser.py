# data_parser.py
# Functions for harmonizing all sorts of input when reading in user data.
#
# Florian P. Bayer - 2024
#

import numpy as np
import pandas as pd

from .search_engine_outputs.MaxQuant import MaxQuantMap
from .search_engine_outputs.DIANN import DiannMap
from .search_engine_outputs.ProteomeDiscoverer import PDMap
from .search_engine_outputs.MSFragger import FraggerMap
from . import user_interface as ui


def clean_modified_sequence(mod_seq):
    """
    Cleans up the modified sequence. Columns-wise operation is 10x faster than apply.

    Parameters
    ----------
    mod_seq : pd.Series(<seqs>)

    Returns
    -------
    mod_seq : pd.Series(<seqs>)
    """
    mappings = {
        r'_': '',
        r'\(\)': '',
        r'\(Acetyl \(Protein N-term\)\)':  '(ac)',
        r'M\(Oxidation \(M\)\)': 'M',
        r'M\(ox\)': 'M',
        r'pS': 'S(ph)',
        r'pT': 'T(ph)',
        r'pY': 'Y(ph)',
        r'\(Phospho \(STY\)\)': '(ph)',
        r'\(Acetyl \(K\)\)': '(ac)',
        r'\(GG \(K\)\)': '(ub)',
    }
    for old, new in mappings.items():
        mod_seq = mod_seq.str.replace(old, new, regex=True)
    return mod_seq


def clean_rows(df):
    """
    Cleans unwanted rows such as Contaminants and Decoys.
    If Columns are not present no error is thrown. The index is resetted after this step.

    Parameters
    ----------
    df : pd.DataFrame
        Data with columns that should be removed.

    Returns
    -------
    df : pd.DataFrame
    """
    if 'Contaminant' in df.columns:
        df = df[~df['Contaminant']]
    if 'Decoy' in df.columns:
        df = df[~df['Decoy']]
    df.reset_index(drop=True, inplace=True)
    return df


def aggregate_duplicates(df, keys, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    """
    Aggregates columns such that key columns are unique afterwards. Depending on the column type different aggegation procedures can be performed.

    df : pd.DataFrame
        input data
    keys : array-like
        column names that should be used to create a unique index
    sum_cols : array-like, optional
        column names where duplicated entries will be summed up (cols of dtype numeric)
    first_cols : array-like, optional
        column names where the first row will be used for duplicated entries (cols of dtype categorical)
    max_cols : array-like, optional
        column names where the max value will be used for duplicated entries (cols of dtype numeric)
    min_cols : array-like, optional
        column names where the min value will be used for duplicated entries (cols of dtype numeric)
    concat_cols : array-like, optional
        column names where all values will be concatenated for duplicated entries (cols of dtype object)

    Returns
    -------
    df : pd.DataFrame
    """
    # Double check that expected columns are present
    ui.verify_columns_exist(df, columns=keys + sum_cols + first_cols + max_cols + min_cols + concat_cols)

    # Create a group by objected that will be re-used a few times later
    grouped_df = df.groupby(keys)

    # Count & Sum the duplicates, then merge results to new_df
    grouped_count = grouped_df.size().to_frame().rename({0: 'N duplicates'}, axis=1)
    grouped_sum = grouped_df[sum_cols].sum(min_count=1)
    new_df = pd.merge(left=grouped_count, right=grouped_sum, left_index=True, right_index=True)

    # use the first element of the group
    if first_cols:
        group_first = grouped_df[first_cols].first()
        new_df = pd.merge(left=new_df, right=group_first, left_index=True, right_index=True)

    # use the max value of the group
    if max_cols:
        group_max = grouped_df[max_cols].max()
        new_df = pd.merge(left=new_df, right=group_max, left_index=True, right_index=True)

    # use the min value of the group
    if min_cols:
        group_min = grouped_df[min_cols].min()
        new_df = pd.merge(left=new_df, right=group_min, left_index=True, right_index=True)

    # concatenate all elements of the group
    if concat_cols:
        df[concat_cols] = df[concat_cols].replace(np.nan, '').astype(str)
        for c_col in concat_cols:
            df_concat = grouped_df[c_col].apply(';'.join)
            new_df = pd.merge(left=new_df, right=df_concat, left_index=True, right_index=True)

    # Resort the columns
    df = new_df[['N duplicates'] + first_cols + concat_cols + max_cols + min_cols + sum_cols]
    df.reset_index(inplace=True)
    return df


#
# DIANN
#


def load_diann_lqf_proteins(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = DiannMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_intensity_columns(df.columns)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df

#
# MaxQuant
#


def load_mq_tmt_peptides(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = MaxQuantMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_tmt_columns(df.columns)
    df = Mapper.map_indicator_values(df)
    df['Modified sequence'] = clean_modified_sequence(df['Modified sequence'])
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


def load_mq_tmt_proteins(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = MaxQuantMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.rename(columns=MaxQuantMap.clean_protein_group_tmt_cols, inplace=True)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_tmt_columns(df.columns)
    df = Mapper.map_indicator_values(df)
    if 'Peptides' in df.columns:
        df = df[df['Peptides'] >= 2]
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


def load_mq_lfq_peptides(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = MaxQuantMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df = Mapper.map_indicator_values(df)
    df = Mapper.restructure_lfq_evidence(df)
    df.columns = Mapper.rename_intensity_columns(df.columns)
    df['Modified sequence'] = clean_modified_sequence(df['Modified sequence'])
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


def load_mq_lqf_proteins(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = MaxQuantMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_lfq_columns(df.columns)
    df = Mapper.map_indicator_values(df)
    if 'Peptides' in df.columns:
        df = df[df['Peptides'] >= 2]
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


#
# Proteome Discoverer
#


def load_pd_tmt_peptides(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = PDMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_tmt_columns(df.columns)
    df = Mapper.create_mod_sequence(df)
    df['Modified sequence'] = clean_modified_sequence(df['Modified sequence'])
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


def load_pd_tmt_proteins(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    raise NotImplementedError()


def load_pd_lqf_peptides(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    raise NotImplementedError()


def load_pd_lqf_proteins(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = PDMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_intensity_columns(df.columns)
    if 'Peptides' in df.columns:
        df = df[df['Peptides'] >= 2]
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


#
# MSFragger
#


def load_fragger_tmt_peptides(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = FraggerMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_tmt_columns(df.columns)
    df = Mapper.create_mod_sequence(df)
    df['Modified sequence'] = clean_modified_sequence(df['Modified sequence'])
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


def load_fragger_tmt_proteins(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = FraggerMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_tmt_columns(df.columns)
    if 'Peptides' in df.columns:
        df = df[df['Peptides'] >= 2]
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


def load_fragger_lqf_peptides(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = FraggerMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_lfq_columns(df.columns)
    df = Mapper.create_mod_sequence(df)
    df['Modified sequence'] = clean_modified_sequence(df['Modified sequence'])
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


def load_fragger_lqf_proteins(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = FraggerMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_lfq_columns(df.columns)
    if 'Peptides' in df.columns:
        df = df[df['Peptides'] >= 2]
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df

#
# Generic
#


def load_generic(path, unique_col, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    # Load
    df = pd.read_csv(path, sep='\t', low_memory=False)
    # A unique column is a requirement for generic upload
    if unique_col not in df.columns:
        raise ValueError(f'The input file must contain a <{unique_col}> column. Please add to the input file.')
    df = aggregate_duplicates(df, keys=[unique_col], sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


def load_generic_peptide_format(path):
    # Load
    df = pd.read_csv(path, sep='\t', low_memory=False)
    return df


def load_generic_protein_format(path):
    # Load
    df = pd.read_csv(path, sep='\t', low_memory=False)
    return df


def load(config):
    """
    Load the input data. Depending on the particular data type use different parser functions.
    The different parsers will return a unified data table for downstream analysis.
    """
    # toml parameters
    path = config['Paths'].get('input_file')
    measurement_type = config['Experiment'].get('measurement_type', 'OTHER').upper()  # <LFQ|TMT|DIA|OTHER>
    data_type = config['Experiment'].get('data_type', 'OTHER').upper()  # <PEPTIDE|PROTEIN|OTHER>
    search_engine = config['Experiment'].get('search_engine', 'OTHER').upper()  # <MAXQUANT|DIANN|OTHER>
    search_engine_version = config['Experiment'].get('search_engine_version', '0.0.0')  # search engine version
    experiments = config['Experiment'].get('experiments')
    ui.message(f' * Loading data file {path}.')

    # columns
    raw_cols = [f'Raw {e}' for e in experiments]

    if (measurement_type == 'DIA') and (search_engine == 'DIANN') and (data_type == 'PROTEIN'):
        unique_cols = ['Genes']
        df = load_diann_lqf_proteins(path, search_engine_version, unique_cols=unique_cols, sum_cols=raw_cols)
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']
        return df

    elif (measurement_type == 'LFQ') and (search_engine == 'MAXQUANT') and (data_type == 'PROTEIN'):
        # TODO: make this to dictionary
        unique_cols = ['Genes', 'Proteins']
        max_cols = ['Score']
        sum_cols = raw_cols + ['Peptides']
        df = load_mq_lqf_proteins(path, search_engine_version, unique_cols=unique_cols, sum_cols=sum_cols, max_cols=max_cols)
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']
        return df

    elif (measurement_type == 'LFQ') and (search_engine == 'PD') and (data_type == 'PROTEIN'):
        # TODO: make this to dictionary
        unique_cols = ['Proteins']
        max_cols = ['Score']
        sum_cols = raw_cols + ['Peptides']
        df = load_pd_lqf_proteins(path, search_engine_version, unique_cols=unique_cols, sum_cols=sum_cols, max_cols=max_cols)
        if 'Genes' not in df.columns:
            df['Genes'] = df['Proteins']
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']
        return df

    elif (measurement_type == 'LFQ') and (search_engine == 'MSFRAGGER') and (data_type == 'PROTEIN'):
        unique_cols = ['Proteins', 'Genes']
        sum_cols = raw_cols + ['Peptides']
        df = load_fragger_lqf_proteins(path, search_engine_version, unique_cols=unique_cols, sum_cols=sum_cols)
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']
        return df

    elif (measurement_type == 'LFQ') and (search_engine == 'MAXQUANT') and (data_type == 'PEPTIDE'):
        # TODO: make this to dictionary
        unique_cols = ['Modified sequence']
        first_cols = ['Genes', 'Proteins']
        max_cols = ['Score']
        df = load_mq_lfq_peptides(path, search_engine_version, unique_cols=unique_cols, first_cols=first_cols, sum_cols=raw_cols, max_cols=max_cols)
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']
        return df

    elif (measurement_type == 'LFQ') and (search_engine == 'MSFRAGGER') and (data_type == 'PEPTIDE'):
        # TODO: make this to dictionary
        unique_cols = ['Modified sequence']
        first_cols = ['Proteins', 'Genes']
        max_cols = []
        df = load_fragger_lqf_peptides(path, search_engine_version, unique_cols=unique_cols, sum_cols=raw_cols, first_cols=first_cols, max_cols=max_cols)
        if 'Genes' not in df.columns:
            df['Genes'] = df['Proteins']
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']
        return df

    elif (measurement_type == 'TMT') and (search_engine == 'MAXQUANT') and (data_type == 'PROTEIN'):
        unique_cols = ['Proteins', 'Genes']
        max_cols = ['Score']
        sum_cols = raw_cols + ['Peptides']
        df = load_mq_tmt_proteins(path, search_engine_version, unique_cols=unique_cols, sum_cols=sum_cols, max_cols=max_cols)
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']
        return df

    elif (measurement_type == 'TMT') and (search_engine == 'MSFRAGGER') and (data_type == 'PROTEIN'):
        unique_cols = ['Proteins', 'Genes']
        sum_cols = raw_cols + ['Peptides']
        df = load_fragger_tmt_proteins(path, search_engine_version, unique_cols=unique_cols, sum_cols=sum_cols)
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']
        return df

    elif (measurement_type == 'TMT') and (search_engine == 'MAXQUANT') and (data_type == 'PEPTIDE'):
        # TODO: make this to dictionary
        unique_cols = ['Modified sequence']
        first_cols = ['Genes', 'Proteins']
        max_cols = ['Score']
        df = load_mq_tmt_peptides(path, search_engine_version, unique_cols=unique_cols, sum_cols=raw_cols, first_cols=first_cols, max_cols=max_cols)
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']
        return df

    elif (measurement_type == 'TMT') and (search_engine == 'PD') and (data_type == 'PEPTIDE'):
        # TODO: make this to dictionary
        unique_cols = ['Modified sequence']
        first_cols = ['Proteins']
        max_cols = []
        df = load_pd_tmt_peptides(path, search_engine_version, unique_cols=unique_cols, sum_cols=raw_cols, first_cols=first_cols, max_cols=max_cols)
        if 'Genes' not in df.columns:
            df['Genes'] = df['Proteins']
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']
        return df

    elif (measurement_type == 'TMT') and (search_engine == 'MSFRAGGER') and (data_type == 'PEPTIDE'):
        # TODO: make this to dictionary
        unique_cols = ['Modified sequence']
        first_cols = ['Proteins', 'Genes']
        max_cols = []
        df = load_fragger_tmt_peptides(path, search_engine_version, unique_cols=unique_cols, sum_cols=raw_cols, first_cols=first_cols, max_cols=max_cols)
        if 'Genes' not in df.columns:
            df['Genes'] = df['Proteins']
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']
        return df

    elif (measurement_type == 'TMT') and (search_engine == 'OTHER') and (data_type == 'PEPTIDE'):
        df = load_generic_peptide_format(path)
        if 'Name' not in df.columns:
            if 'Genes' in df.columns:
                df['Name'] = df['Genes']
            else:
                df['Name'] = df.index.values.copy()
        return df

    elif (measurement_type == 'TMT') and (search_engine == 'OTHER') and (data_type == 'PROTEIN'):
        df = load_generic_protein_format(path)
        if 'Name' not in df.columns:
            if 'Genes' in df.columns:
                df['Name'] = df['Genes']
            else:
                df['Name'] = df.index.values.copy()
        return df

    elif (measurement_type == 'OTHER') and (search_engine == 'OTHER') and (data_type == 'OTHER'):
        unique_col = 'Name'
        df = load_generic(path, unique_col=unique_col, sum_cols=raw_cols)
        return df

    else:
        msg = f'The combination of measurement_type = "{measurement_type}", data type = "{data_type}", and  search_engine = "{search_engine}" is currently not supported.'
        raise NotImplementedError(msg)
