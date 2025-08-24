# toolbox.py
# Functions needed everywhere.
#
# Florian P. Bayer - 2025
#


# Imports
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
import multiprocessing
import functools
from collections import defaultdict

import numpy as np
import pandas as pd
import tqdm


def parallelize_dataframe(df, func, return_cols, func_kwargs={}, n_cores=1, sorted=True, **kwargs):
    """
    This function enables parallelization of a function on a data frame in a row-wise fashion.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that stores the data.
    func : object
        A function that works on the data frame in a row-wise fashion.
        The apply function must return a tuple of values.
        The first value is expected to return the index. All other values must match to return_cols.
    return_cols : list
        A list of column names. One for each output value of the func.
    func_kwargs : dict
        keyword arguments to pass to the function
    n_cores : int
        number of cores to use.
    sorted : bool, optional
        If the return DataFrame should be sorted by the index order of the input DataFrame.
    kwargs

    Returns
    -------
    out_df : pd.DataFrame (shape: df.index * return_cols)
        Processed DataFrame by the function.
    """
    # Perform the parallelized computation by mapping func with kwargs on rows.
    # The imap_unordered returns a lazy generator from the multiprocessing.pool.
    # The lazy result is then wrapped into tqdm for a progress bar.
    # Parallel execution happens when DataFrame is built. The first return value is expected to be the index.
    rows = (row for _, row in df.iterrows())
    pool = multiprocessing.Pool(processes=n_cores)
    try:
        results = pool.imap_unordered(functools.partial(func, **func_kwargs), rows)
        results = (r for r in tqdm.tqdm(results, total=len(df)))  # add tqdm to generator
        out_df = pd.DataFrame.from_records(results, index='index', columns=['index'] + list(return_cols)) # build result DF
    # Kill all workers right away with 'CRTL-C' and exit the program.
    except KeyboardInterrupt:
        pool.terminate()
        raise SystemExit()
    # Any other exception triggered will terminate and pass Exception.
    except Exception:
        pool.terminate()
        raise
    #  When done, close the pool and
    else:
        pool.close()
    # Always wait for the worker processes to terminate.
    finally:
        pool.join()

    # Sort index to input index
    # This becomes necessary because imap_unordered returns in an arbitrary order.
    out_df.index.name = df.index.name
    if sorted:
        out_df = out_df.loc[df.index]
    return out_df


def build_drug_log_concentrations(steps, scale=1, dmso_offset=1e3):
    """
    This function takes a sequence of drug steps and its corresponding scale and calculates the respective
    concentration numpy array. The DMSO (0) will be corrected to a n times smaller number than the smallest value.
    This is necessary for log-step transformations later on. Default shift is 1000.

    Parameters
    ----------
    steps : array-like
        The drug steps in real space. Order does not matter
    scale : numeric
        The drug scaling factor
    dmso_offset : numeric
        The dmso offset to display the dmso in log space. By default 1000 (=3 orders of magnitude)

    Returns
    -------
    log_steps : array-like
        The drug log steps in the original order
    """
    # Return empty array if empty input
    if len(steps) == 0:
        return np.array([])
    # Else calculate the log steps with defined dmso offset based on input
    steps = np.array(steps)
    steps[steps == 0] = min(steps[steps != 0]) / dmso_offset
    log_steps = np.log10(steps * scale)
    return log_steps


def build_col_names(col_name, iter_list):
    """
    Build the names for the reporter columns
    """
    return np.array([col_name.format(i) for i in iter_list])


def roundup(x):
    """
    Rounds up to first significant digit
    """
    if x == 0.0:
        return 0.0
    elif x < 0.0:
        return - rounddown(abs(x))
    n_digits = np.floor(np.log10(abs(x)))
    rounded = np.ceil(x / 10**n_digits) * 10**n_digits
    return rounded


def rounddown(x):
    """
    Rounds down to first significant digit
    """
    if x == 0.0:
        return 0.0
    elif x < 0.0:
        return - roundup(abs(x))
    n_digits = np.floor(np.log10(abs(x)))
    rounded = np.ceil(x / 10**(n_digits+1)) * 10**n_digits
    return rounded


def aggregate_xy(x, y, agg_fun=np.nanmean):
    """
    Aggregates the y values using an aggregation function based on x grouping.

    Parameters
    ----------
    x : array-like
        List of x values used for grouping.
    y : array-like
        List of y values that are aggregated.
    agg_fun : function
        Aggregation function that should be applied to y for each group of x. By default np.nanmean.

    Returns
    -------
    x_agg : array-like
        Groups of x.
    y_agg : array-like
        aggregated values of y based on x groups.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        dd = defaultdict(list)
        for x_i, y_i in zip(x, y):
            dd[x_i].append(y_i)
        x_agg = np.fromiter(dd.keys(), dtype=float)
        y_agg = np.fromiter(map(agg_fun, dd.values()), dtype=float)
    return x_agg, y_agg
