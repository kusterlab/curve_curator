# data_simulator.py
# Simulates random curves given the H0 is True meaning that x & y are independent.
#
# Florian P. Bayer - 2024
#

import numpy as np
import pandas as pd
from scipy import stats

from . import user_interface as ui
from . import toolbox as tool
from .models import LogisticModel

from tqdm.autonotebook import tqdm
tqdm.pandas()


# The intrinsic variance models. Estimated from decryptM data.
DEFAULT_VARIANCE_MODEL = stats.f(dfn=11, dfd=11, loc=0.035, scale=0.066)


def simulate_h0_dataset(cols, n_curves):
    """
    Simulates n random curves under the H0 hypothesis

    Parameters
    ----------
    cols : list of column names
        The Raw column names that should be simulated.
    n_curves : int
        The number of curves that should be simulated.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame()
    n_doses = len(cols)

    # Sample n curves with each having a different intrinsic variance
    df['STD'] = DEFAULT_VARIANCE_MODEL.rvs(size=n_curves)
    df.index.name = 'Name'

    # Sample observations for each curve which is y_i = 1 * Error ~ N(1, sts)
    values = np.full((n_curves, n_doses), np.nan)
    for i, s in tqdm(enumerate(df['STD']), total=n_curves):
        values[i, :] = stats.norm.rvs(loc=1, scale=s, size=n_doses)
    df[cols] = values

    # Clip Values to be in good range
    df = df.clip(lower=1/1000, upper=1000)
    return df


def estimate_noise_distribution(df, cols, x):
    """
    Estimates the experimental noise distribution from the curve data.

    Parameters
    ----------
    df : pd.DataFrame
        real target data with curve fits. Needs at least the model columns after the curve fit and the ratio observation data.
    cols : list of column names
        The ratio column names
    x : array-like
        drug concentrations in log-space

    Returns
    -------
    array of noise estimates
    """
    # unpack for faster execution
    pec50 = df['pEC50'].values
    cslope = df['Curve Slope'].values
    cfront = df['Curve Front'].values
    cback = df['Curve Back'].values
    Y = df[cols].values
    n = len(cols) - df[cols].isna().sum(axis=1).values
    # Loop and calculate initial noise level
    # calculate noise with dofs as any curve fit will remove variance which is then missing for the correct estimate
    noise = np.empty_like(n, dtype=float)
    for i in range(len(noise)):
        LM = LogisticModel(pec50=pec50[i], slope=cslope[i], front=cfront[i], back=cback[i])
        _, dfd = LM.get_dofs(n[i])
        noise[i] = np.sqrt(1 / dfd * LM.calculate_sum_squared_residuals(x, Y[i, :]))
    return noise


def simulate_decoys(n_decoys, cols, empirical_noise=None):
    """
    Simulates decoys based on empirical noise distribution

    Parameters
    ----------
    n_decoys : int
        number of decoys to simulate
    cols : list of strings
        column names for decoy raw values
    empirical_noise : array-like
        list of noise values

    Returns
    -------
    pd.DataFrame with simulated decoys
    """
    n_doses = len(cols)

    # Draw decoy noise distribution from empirical data and make it more robust against outliers in the empirical noise distribution
    max_noise = np.quantile(empirical_noise[np.isfinite(empirical_noise)], q=0.99) * 1.1
    min_noise = np.quantile(empirical_noise[np.isfinite(empirical_noise)], q=0.01) / 1.1
    sampled_noise = np.random.choice(empirical_noise[(empirical_noise >= min_noise) & (empirical_noise <= max_noise)], size=n_decoys)

    # Sample decoy curves which is y_i = 1 * Error ~ N(1, noise)
    decoys = np.full((n_decoys, n_doses), np.nan, dtype=float)
    for i, noise in tqdm(enumerate(sampled_noise), total=n_decoys):
        decoy_noise = min_noise
        while decoy_noise <= min_noise:
            decoy = stats.norm.rvs(loc=1, scale=noise, size=n_doses)
            decoy_noise = decoy.std()
            noise *= 1.01  # guarantee to exit while loop
        decoys[i, :] = decoy

    # Typecast to data frame
    decoys_df = pd.DataFrame.from_records(decoys, columns=cols)
    decoys_df.index.name = 'Name'
    decoys_df.index = 'Decoy_' + decoys_df.index.astype(str)

    # Clip Values to be in a good range
    decoys_df = decoys_df.clip(lower=1/1000, upper=1000)
    decoys_df.reset_index(inplace=True)
    return decoys_df


def sample(config, n):
    """
    main function
    """
    experiments = np.array(config['Experiment']['experiments'])
    cols = [f'Raw {e}' for e in experiments]

    # Simulate curves
    ui.message(f' * Simulating {n} random curves:')
    df = simulate_h0_dataset(cols, n)

    # Save files
    df.to_csv(config['Paths']['input_file'], sep='\t')
    ui.message(f' * Simulation done.')


def get_decoys(df, config):
    """
    main function
    """
    # get relevant parameters
    experiments = np.array(config['Experiment']['experiments'])
    raw_cols = [f'Raw {e}' for e in experiments]
    ratio_cols = [f'Ratio {e}' for e in experiments]
    decoy_ratio = config['F Statistic']['decoy_ratio']
    drug_log_concs = tool.build_drug_log_concentrations(np.array(config['Experiment']['doses']), config['Experiment']['dose_scale'])

    # Estimating noise distribution for decoys
    ui.message(f' * Estimating noise distribution based on target curves.')
    empirical_noise = estimate_noise_distribution(df, cols=ratio_cols, x=drug_log_concs)
    n = int(decoy_ratio * len(empirical_noise))
    if n < 1e4:
        ui.warning(f' * Less then 10k decoys may result in inaccurate FDR estimations. Consider increasing the decoy_ratio parameter.')

    # Simulate curves
    ui.message(f' * Simulating {n} decoys based on target noise distribution:')
    df = simulate_decoys(n_decoys=n, cols=raw_cols, empirical_noise=empirical_noise)

    # Save files
    ui.message(f' * Decoy simulation done.')
    return df
