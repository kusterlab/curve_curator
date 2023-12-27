# thresholding.py
# Apply SAM 2D thresholds to the dose-response curves to identify significant curves.
#
# Florian P. Bayer - 2024
#

from scipy.stats import f as f_distribution
import statsmodels.api as sm
import pandas as pd
import numpy as np

from . import toolbox as tool
from . import user_interface as ui
from .models import LogisticModel


def get_s0(fc_lim, alpha, dfn, dfd, loc=0, scale=1, two_sided=False):
    """
    Calculates the s0 value given a log2 fold change limit and an alpha value.
    This is based on the two-tailed SAM test analysis transferred two F-test.

    Parameters
    ----------
    fc_lim : float or array of floats
        fold-change limit asymptote (x) in log2.
    alpha : float or array of floats
        alpha threshold limit asymptote (y). Between 0 and 1.
    dfn : float or array of floats
        degrees of freedom of nominator for the f-distribution.
    dfd : float or array of
        degrees of freedom of denominator for the f-distribution.
    loc : float or array of floats
        location parameter for the f-distribution.
    scale : float or array of floats
        scaling parameter for the f-distribution.
    two_sided : bool, optional
        if a two-sided test is performed. Default is False.

    Returns
    -------
    s0 : float
        fudge factor s0
    """
    # Convert to vectorized form and check input
    fc_lim, alpha = np.asarray(fc_lim), np.asarray(alpha)
    if np.any((alpha < 0) | (alpha > 1)):
        raise ValueError(f'alpha value(s) must be between 0 and 1.')
    # Using the F = T**2 equality for 2 groups (max vs. min plateau), SAM analysis can be performed with f-distribution
    if two_sided:
        alpha_lim = np.sqrt(f_distribution.ppf(1 - (alpha / 2), dfn=dfn, dfd=dfd, loc=loc, scale=scale))
    else:
        alpha_lim = np.sqrt(f_distribution.ppf(1 - alpha, dfn=dfn, dfd=dfd, loc=loc, scale=scale))
    # Calculate S0 in vectorized form
    alpha_lim = np.asarray(alpha_lim)
    alpha_lim[alpha_lim < 0] = 0.0
    s0 = abs(fc_lim) / alpha_lim
    return s0


def get_fclim(s0, alpha, dfn, dfd, loc=0, scale=1, two_sided=False):
    """
    Calculates the +- log2 fold change limit (asymptote) given a s0 value and an alpha value.
    This is based on the two-tailed SAM test analysis transferred two F-test.

    Parameters
    ----------
    s0 : float or array of floats
        fudge factor, which determines the transition between both fold-change and p-value asymptotes.
    alpha : float or array of floats
        alpha threshold limit asymptote (y). Between 0 and 1.
    dfn : float or array of floats
        degrees of freedom of nominator for the f-distribution.
    dfd : int or array of floats
        degrees of freedom of denominator for the f-distribution..
    loc : float or array of floats
        location parameter for the f-distribution.
    scale : float or array of floats
        scaling parameter for the f-distribution.
    two_sided : bool, optional
        if a two-sided test is performed. By default False.

    Returns
    -------
    -fc_lim, fc_lim : (float, float)
        +- log2 fold change limits
    """
    # Convert to vectorized form and check input
    s0, alpha = np.asarray(s0), np.asarray(alpha)
    if np.any((alpha < 0) | (alpha > 1)):
        raise ValueError(f'alpha value(s) must be between 0 and 1.')
    # Using the F = T**2 equality for 2 groups (max vs. min plateau), SAM analysis can be performed with f-distribution
    if two_sided:
        alpha_lim = np.sqrt(f_distribution.ppf(1 - (alpha / 2), dfn=dfn, dfd=dfd, loc=loc, scale=scale))
    else:
        alpha_lim = np.sqrt(f_distribution.ppf(1 - alpha, dfn=dfn, dfd=dfd, loc=loc, scale=scale))
    # Calculate +/- fc_lim pairs in vectorized form
    alpha_lim = np.asarray(alpha_lim)
    fc_lim = abs(alpha_lim * s0)
    return -fc_lim, fc_lim


def sam_correction(f_values, curve_fold_change, s0):
    """
    Multiple testing correction of F_values using the SAM principle applied to does response curves and F statistic.

    Parameters
    ----------
    f _values : array-like
        Curve curator F-values.
    curve_fold_change : array-like
        Curve fold changes to the corresponding f-values.
    s0 : float
        fudge factor, which determines the transition between both fold-change and p-value asymptotes.

    Returns
    -------
    f_values_adjusted

    Comments
    --------
    The classical SAM adjustment for t-tests uses the formula: [eq.1] t_adj = fc / (std + s0).
    Transforming this equation yields: [eq.2] 1/t_adj = 1/t_old + s0/fc.
    Basic idea now is to convert the F-values to the t-value equivalents as the curve limits between the two groups of the two plateaus.
    Then, the transformed equation 2 becomes: [eq.3] 1/sqrt(F_adj) = 1/sqrt(F) + s0/fc
    Solving again for F_adj results in the final function: [eq.4]  F_adj = 1 / ((1/sqrt(F)) + (s0/fc))**2
    """
    # Convert to vectorized form
    f_values, curve_fold_change, s0 = np.asarray(f_values), np.asarray(curve_fold_change), np.asarray(s0)
    # Calculate s0-adjusted f-values
    with np.errstate(divide='ignore'):
        f_values_adjusted = 1 / ((1 / np.sqrt(f_values)) + (s0 / abs(curve_fold_change)))**2
    return f_values_adjusted


def map_fc_to_pvalue_cutoff(x, alpha, s0, dfn, dfd, loc=0, scale=1, two_sided=False):
    """
    This function maps input fold changes to the respective p-values given statistic type, chosen alpha value,
    fudge factor s0, and degrees of freedom. It is based on the SAM test with some modifications (see comments).

    Parameters
    ----------
    x : pd.Series
        log2 fold change values
    alpha : float
        alpha threshold limit asymptote (y). Between 0 and 1.
    s0 : float
        fudge factor, which determines the transition between both fold-change and p-value asymptotes.
    dfn : int
        degrees of freedom of nominator for the f-distribution.
    dfd : int
        degrees of freedom of denominator for the f-distribution..
    loc : float
        location parameter for the f-distribution.
    scale : float
        scaling parameter for the f-distribution.
    two_sided : bool, optional
        if a two-sided test is performed. By default False.

    Returns
    -------
    y : pd.Series
        p-value cutoffs for each input log2-fold change x-value.

    Comments
    --------
    Adapted from:
    https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/pmic.201600132
    R Code in the supplement
    FPB addition: use the f_distribution for calculation of a f-statistic under assumption F = T**2 for two group analysis
    """
    if (len(x) == 0) or type(pd.Series([0])) is not pd.Series:
        raise ValueError(f'Fold change array needs to be a pd.Series object with at least 1 value.')
    if not 0 <= alpha <= 1:
        raise ValueError(f'Alpha value must be between 0 and 1, but it was {alpha}.')
    # Using the F = T**2 equality for 2 groups (max vs. min plateau infinity argument), SAM analysis can be performed with f-distribution
    if two_sided:
        alpha_lim = np.sqrt(f_distribution.ppf(1 - (alpha / 2), dfn=dfn, dfd=dfd, loc=loc, scale=scale))
    else:
        alpha_lim = np.sqrt(f_distribution.ppf(1 - alpha, dfn=dfn, dfd=dfd, loc=loc, scale=scale))

    # New positional limits with s0
    pos_lim = alpha_lim * s0
    neg_lim = -alpha_lim * s0

    # Mask with 0 edge case
    pos = x > pos_lim
    neg = x < neg_lim
    if pos_lim == neg_lim:
        pos = x >= pos_lim

    # Calculate the fudge-modified values
    x_pos = x[pos]
    x_neg = x[neg]
    x_none = x[(~pos) & (~neg)]

    d_pos = x_pos / alpha_lim - s0
    d_pos = (s0 / d_pos)
    d_pos = alpha_lim * (1 + d_pos)

    d_neg = x_neg / (-alpha_lim) - s0
    d_neg = (s0 / d_neg)
    d_neg = alpha_lim * (1 + d_neg)

    # Calculate to p-values.
    # Revert to F-test with F = T**2
    # Log survival function to have more accurate p value calculations: - np.log10[(1 - dist.cdf(x, dfn, dfd))]
    y_pos = - f_distribution.logsf(d_pos**2, dfn=dfn, dfd=dfd, loc=loc, scale=scale) * np.log10(np.e)
    y_neg = - f_distribution.logsf(d_neg**2, dfn=dfn, dfd=dfd, loc=loc, scale=scale) * np.log10(np.e)
    if two_sided:
        # Two sided p values are multiplied by two: - np.log10[(1 - dist.cdf(x, deg)) * 2]
        y_pos -= np.log10(2)
        y_neg -= np.log10(2)

    # Combine arrays for output and ensure non-negative values and convert nan to max_prob as its the least likely (for the 0 edge case for d_p/n)
    y_none = np.full(shape=len(x_none), fill_value=np.inf)
    y = pd.concat([pd.Series(y_neg, index=x_neg.index), pd.Series(y_none, index=x_none.index), pd.Series(y_pos, index=x_pos.index)])
    y = y.clip(lower=0, upper=None)
    y = y.replace(np.nan, max(y))
    return y[x.index]


def correct_pvalues(pvalues, alpha=0.01, method='fdr_bh'):
    """
    FDR Correction of an unsorted array of p_values using sm.stats.multipletests.

    Parameters
    ----------
    pvalues: array-like
        Not-FDR corrected p-values
    alpha: float, optional
        The family-wise error rate of the data set, default 1% (0.01)
    method: string, optional
        The FDR Method to apply from statsmodels:
        'bonferroni' = one-step correction
        'sidak' = one-step correction
        'holm-sidak' = step down method using Sidak adjustments
        'holm' = step-down method using Bonferroni adjustments
        'simes-hochberg' = step-up method (independent)
        'hommel' = closed method based on Simes tests (non-negative)
        'fdr_bh' = Benjamini/Hochberg (non-negative)
        'fdr_by' = Benjamini/Yekutieli (negative)
        'fdr_tsbh' = two stage fdr correction (non-negative)
        'fdr_tsbky' = two stage fdr correction (non-negative)

    Returns
    -------
    pvalues_out: array-like
        FDR corrected p-values

    Notes
    -----
    Just a wrapper function for the statsmodel, which will fail when NA's are passed to the function.
    """
    mask = np.isfinite(pvalues)
    pvalues_out = np.full(len(pvalues), np.nan)
    reject, pvals_corrected, alphacSidak, alphacBonf = sm.stats.multipletests(pvalues[mask], alpha=alpha, method=method)
    pvalues_out[mask] = pvals_corrected
    return pvalues_out


def define_regulated_curves(df, cut_col, cut_value, fc_lim, not_rmse_limit, not_cut_limit, quality_min):
    """
    Find regulated curves and add a new column <Regulation> to the input df.
    Regulation can have three values: significant up, significant down, clearly not.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cut_col : str
        Column name that is used to cut with the the cut value.
   cut_value : float
        The threshold value that is used to cut significant curves.
    fc_lim : float
        The log2 fold change limit.
    not_rmse_limit : float
        The maximal root mean squared error limit of the null model to be still considered not regulated.
    not_cut_limit : float
        The maximal value limit of a curve to be still considered not regulated.
    quality_min : float
        The minimum data quality required for a curve to have sufficient quantification.

    Returns
    -------
    df : pd.DataFrame
    """
    # Define the masks for significant up and down
    p_mask = df[cut_col] >= cut_value
    down_mask = df['Curve Fold Change'] < 0
    up_mask = df['Curve Fold Change'] > 0
    effect_mask = df['Curve Fold Change'].abs() >= fc_lim

    # Not regulated curves require: not too much noise, not too much deviation to control, not too likely a log-logistic curve
    fc_range = -abs(fc_lim)/2, abs(fc_lim)/2
    not_regulated_mask = (df['Null RMSE'] <= not_rmse_limit) & (np.log2(df['Null Model'])).between(*fc_range) #& (df['cut_col'] <= not_cut_limit)

    # Add labels
    df['Curve Regulation'] = ''
    df.loc[~p_mask & not_regulated_mask, 'Curve Regulation'] = 'not'
    df.loc[p_mask & effect_mask & up_mask, 'Curve Regulation'] = 'up'
    df.loc[p_mask & effect_mask & down_mask, 'Curve Regulation'] = 'down'
    df['Curve Regulation'] = df['Curve Regulation'].replace('', np.nan)

    # Apply the min_signal filter by removing potential regulations that are below the min signal threshold.
    df.loc[df['Signal Quality'] <= quality_min, 'Curve Regulation'] = np.nan
    return df


def calculate_qvalue(df, sort_cols, sort_ascendings, decoy_col, q_col_name='Curve q-value'):
    """
    Calculate the q_values based on target decoy approach. the q-value gives the expected pFDR obtained by rejecting the null hypothesis
    for any result with an equal or smaller q-value. The q value will be added to the given data frame with the q col name.

    Parameters
    ----------
    df : pd.DataFrame
        input data frame with at least the sort cols and a decoy_col.
    sort_cols : list(["col1", "col2", ..])
        list of column names used for sorting. Order matters. First element with highest priority.
    sort_ascendings: list(<bool>, ..)
        list of booleans, corresponding to the sort_cols indicating the direction of sorting.
    decoy_col : str
        the column name of the decoys. This column should be of type bool.
    q_col_name : str, optional
        the column name of the calculated q value, default = 'Curve q-value'

    Returns
    -------
    df : pd.DataFrame
        df with the added q_value column
    """
    # Sort values
    df_sorted = df.dropna(subset=sort_cols).sort_values(by=sort_cols, ascending=sort_ascendings)

    # Calculate q values based on decoy order
    decoys = df_sorted[decoy_col].astype(bool).cumsum()
    targets = df_sorted[decoy_col].astype(bool).apply(np.logical_not).cumsum()
    q = ((decoys + 1) / (targets + 1)).astype(float)
    q = q[::-1].cummin()[::-1]  # Make q values monotonically increasing

    # Adjust q-values for unbalanced target:decoy ratio
    ratio = sum(df[decoy_col] == False) / sum(df[decoy_col] == True)
    q = q * ratio

    # Add to dataframe
    df[q_col_name] = q
    return df


def get_dofs_silently(n, model, optimized):
    """
    Wrapper function to safely calculate dofs even in the event of AssertionError when too few data points per curve exist.
    When assertion error is triggered return nan which will prevent calculation of an relevance score.
    """
    try:
        return model.get_dofs(n, optimized=optimized)
    except ValueError:
        return np.nan, np.nan


def apply_significance_thresholds(df, config):
    """
    main function. Do the thresholding based on the config file.
    """
    # Get parameter from toml file
    drug_doses = np.array(config['Experiment']['doses'])
    control_mask = (drug_doses > 0.0)
    n = control_mask.sum() + 1  # 1 for control data point
    cols_ratio = tool.build_col_names('Ratio {}', config['Experiment']['experiments'][control_mask])
    fc_lim = config['F Statistic']['fc_lim']
    alpha = config['F Statistic']['alpha']
    loc = config['F Statistic']['loc']
    scale = config['F Statistic']['scale']
    two_sided = config['F Statistic']['two_sided']
    optimized_dofs = config['F Statistic']['optimized_dofs']

    # Perform multiple testing correction and cut significant curves
    not_rmse_limit = config['F Statistic']['not_rmse_limit']
    not_p_limit = config['F Statistic']['not_p_limit']
    quality_min = config['F Statistic']['quality_min']
    multiple_testing_method = config['F Statistic']['mtc_method']
    log_alpha = -np.log10(alpha)

    if multiple_testing_method == 'sam':
        ui.message(' * Calculate Relevance Score and apply SAM user thresholds:')

        # Setup the logistic model with correct degrees of freedom to calculate default s0
        model = LogisticModel(slope=config['Curve Fit'].get('slope'), front=config['Curve Fit'].get('front'), back=config['Curve Fit'].get('back'))
        dfn, dfd = model.get_dofs(n, optimized=optimized_dofs)
        dfn, dfd = config['F Statistic'].get('dfn', dfn), config['F Statistic'].get('dfd', dfd)  # overwrites values if present in toml file
        default_s0 = get_s0(fc_lim, alpha, dfn=dfn, dfd=dfd, two_sided=two_sided)
        ui.message(f'   alpha={alpha}, fc_lim={fc_lim}, s0={default_s0:.4f}', end='\n')

        # S0 can be different between different curves when there are missing values in curves. Therefore the s0 calculation must be vectorized.
        valid_values = n - df[cols_ratio].isna().sum(axis=1)
        dofs = pd.DataFrame.from_records(valid_values.apply(get_dofs_silently, model=model, optimized=optimized_dofs), columns=['dfn', 'dfd'])
        dfn, dfd = config['F Statistic'].get('dfn', dofs['dfn']), config['F Statistic'].get('dfd', dofs['dfd'])
        vector_s0 = get_s0(fc_lim, alpha, dfn=dfn, dfd=dfd, two_sided=two_sided)

        # Correct F and p values using the SAM principle to control multiple testing
        df['Curve F_Value SAM Corrected'] = sam_correction(df['Curve F_Value'], df['Curve Fold Change'], s0=vector_s0)
        df['Curve Relevance Score'] = -np.log10(f_distribution.sf(df['Curve F_Value SAM Corrected'], dfn=dfn, dfd=dfd, scale=scale, loc=loc))
        df = define_regulated_curves(df, 'Curve Relevance Score', log_alpha, fc_lim, not_rmse_limit, not_p_limit, quality_min)

    else:
        ui.message(f' * Calculate adjusted p-values using the {multiple_testing_method} method:', end='\n')
        df['Curve P_Value adjusted'] = correct_pvalues(df['Curve P_Value'], alpha=alpha, method=multiple_testing_method)
        df['Curve Log P_Value adjusted'] = -np.log10(df['Curve P_Value adjusted'])
        df = define_regulated_curves(df, 'Curve Log P_Value adjusted', log_alpha, fc_lim, not_rmse_limit, not_p_limit, quality_min)

    return df


def estimate_fdr(target_df, decoy_df, config):
    """
    main function. Estimate the FDR using target-decoy approach for SAM-correction and Relevance Score.
    """
    # Check that only SAM analysis has FDR option
    multiple_testing_method = config['F Statistic']['mtc_method']
    if multiple_testing_method != 'sam':
        ui.warning(f' * FDR estimation is only available for SAM-method and not for adjusted p-values method {multiple_testing_method}!')
        return np.nan

    # Define sort columns for q value and make sure that they are present
    sort_cols = {'Curve Relevance Score': False, 'Curve F_Value SAM Corrected': False}
    for col in sort_cols.keys():
        if col not in target_df or col not in decoy_df:
            del sort_cols[col]

    # Combine target decoy data set but keep it mappable by index
    decoy_df.set_index('Name', inplace=True)
    decoy_df['Decoy'] = True
    df_combined = pd.concat([target_df, decoy_df])
    assert df_combined.index.is_unique
    df_combined['Decoy'] = df_combined['Decoy'].replace(np.nan, False)

    # Calculate q value and add to target
    df_combined = calculate_qvalue(df_combined, sort_cols=list(sort_cols.keys()), sort_ascendings=list(sort_cols.values()), decoy_col='Decoy', q_col_name='Curve q_Value')
    target_df['Curve q_Value'] = df_combined.loc[target_df.index, 'Curve q_Value']
    decoy_df['Curve q_Value'] = df_combined.loc[decoy_df.index, 'Curve q_Value']
    decoy_df.reset_index(inplace=True)

    # Calculate FDR and report
    regulation_mask = target_df['Curve Regulation'].isin({'up', 'down'})
    fdr = target_df[regulation_mask]['Curve q_Value'].max()
    ui.message(f' * Estimated FDR with given user thresholds is: {fdr:.2g}')
    if config['Paths'].get('fdr_file'):
        with open(config['Paths']['fdr_file'], 'w') as out_file:
            out_file.write(f'{fdr:.4g}')
    return fdr
