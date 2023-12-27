# quantification.py
# Functions for processing and fitting dose-response curves.
#
# Florian P. Bayer - 2024
#


# Imports:
import pandas as pd
import numpy as np
from scipy import stats, interpolate

from . import toolbox as tool
from . import user_interface as ui
from .models import MeanModel, LogisticModel

from tqdm.autonotebook import tqdm
tqdm.pandas()


def filter_nans(df, cols, max_missing):
    """
    Filters out rows with too many missing values in the specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame containing at least cols.
    cols : array-like
        A array-like object containing the column labels of some intensity data.
    max_missing : int
        Number of maximal allowed missing values.

    Returns
    -------
    df : pd.DataFrame
    """
    mask = df[cols].replace(0, np.nan).isna().sum(axis=1) <= max_missing
    df = df[mask].copy()
    df = df.reset_index(drop=True)
    return df


def get_imputation_value(df, col, pct=0.005):
    """
    Find a good static imputation value based on given column in data frame.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame containing at least col name.
    col : array-like of str
        A array-like object of column name(s) in the df with values from which a good imputation value is drawn.
    pct : float, optional
        Percentile threshold which is used to find a good value for imputation. By default 0.005.

    Returns
    -------
    value : float
        imputation value.
    """
    value = df[col].mean(axis=1).replace(0, np.nan).dropna().quantile(pct)
    return value


def impute_nans(df, raw_cols, imputation_value, max_imputations):
    """
    Impute intensity values based on low signal cutoff (= single intensity value).
    Values below the imputation value will be updated to imputation value but not counted as imputed value.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame containing at least raw_cols and noise_cols.
    raw_cols : array-like
        A array-like object containing the column labels of the raw intensity data.
    imputation_value : float
        the minimum intensity value in the data set.
    max_imputations : int
        the maximum number of imputations allowed per curve.

    Returns
    -------
    df : pd.DataFrame
    """
    def join_true_positions_from_index(row):
        return ';'.join(map(lambda col_name: col_name.split(" ")[-1], row[row].index))

    # Define the imputation matrix
    imputation_mask = df[raw_cols].isna() | (df[raw_cols] < imputation_value)
    # Make some meta statistic
    df['Imputation N'] = imputation_mask.sum(axis=1)
    df['Imputation Position'] = imputation_mask.apply(lambda row: join_true_positions_from_index(row), axis=1)
    # Update where necessary
    df[raw_cols] = df[raw_cols].clip(lower=imputation_value).replace(np.nan, imputation_value)
    # Filter out if too much imputation was done
    mask = df['Imputation N'] <= max_imputations
    df = df[mask].copy()
    df = df.reset_index(drop=True)
    return df


def normalize_values(df, raw_cols, norm_cols, ref_col=None):
    """
    Median centric normalization of raw_cols using the rows of the reference only.
    The normalized values will be added to the df under the name of norm_cols.
    Missing values in the raw_cols are conserved as NaNs in the norm_cols.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame containing at least raw_cols, ref_col
    raw_cols : array-like
        A array-like object containing the column labels of the raw data
    norm_cols : array-like
        A array-like object containing the column labels of the future normalized data
    ref_col : string, optional
        A string indicating a column name that stores the booleans which will be used as a reference for normalization.
        By default None, meaning that all rows are used to calculate the normalization factors.

    Returns
    -------
    df : pd.DataFrame
        The result data frame with the added norm_cols.
    normalization_factors : array-like
        The applied normalization factors.
    """
    ref_mask = df[ref_col] if ref_col else df.index
    ref_medians = np.log2(df.loc[ref_mask, raw_cols].replace(0, np.nan)).median()
    normalization_factors = ref_medians.mean() - ref_medians
    df[norm_cols] = 2 ** (np.log2(df[raw_cols].replace(0, np.nan)) + normalization_factors)
    df[norm_cols] = df[norm_cols].replace([np.nan, -np.inf, np.inf], 0)
    # Conserve the missing values from raw_cols
    nan_mask = df[raw_cols].isna().copy().rename(columns=dict(zip(raw_cols, norm_cols)))
    df[nan_mask] = np.nan
    return df, normalization_factors


def add_ratios(df, cols, ratio_cols, ref_cols):
    """
    Calculate ratios of cols / ref_col.
    The ratio values will be added to the df under the name of ratio_cols.
    In case of multiple columns the mean of ref_cols is used to calculate ratios.

    Parameters
    ----------
    df : pd.DataFrame
        A data frame containing at least raw_cols, ref_col.
    cols : array-like
        A array-like object containing the column labels of the data.
    ratio_cols : array-like
        A array-like object containing the column labels of the future ratio data.
    ref_cols : array-like
        A array-like object of strings indicating one or multiple column(s) used as a reference for ratio calculations.

    Returns
    -------
    df : pd.DataFrame
        The result data frame with the added ratio_cols.
    """
    df[ratio_cols] = df[cols].div(df[ref_cols].mean(axis=1), axis=0).replace([np.inf], np.nan)
    return df


def build_interpolation_points(x, exclude_low_n=0, exclude_top_n=0, interpolation_size=np.log10(10/3)/2):
    """
    Constructs an interpolated array, where the new elements lay with space of maximum interpolation_size between
    the x elements array.

    Parameters
    ----------
    x : array-like
        Input array with drug concentrations in log space.
    exclude_low_n : int
        The number of elements to exclude from the interpolation at the lower end. By default 1 (=exclude DMSO).
    exclude_top_n : int
        The number of elements to exclude from the interpolation at the higher end. By default 0.
    interpolation_size : float
        The maximum interpolation size before distance between x is halved.
        Default is np.log10(10/3)/2 for half log10 step interpolation.

    Returns
    -------
    x_interpolated : array-like
        An array with added interpolation points that are equidistant smaller than interpolation_size.
    """
    while_counter = 0
    while True:
        # sort x and calculate the x difference to
        x = sorted(x)
        x_diff = np.diff(x[exclude_low_n:len(x) - exclude_top_n])

        # if all smaller then full log step
        if all(x_diff <= interpolation_size):
            return np.array(x)
        # else: half the distance where it is too large
        half_x = list((x[pos + exclude_low_n] + x[pos + exclude_low_n + 1]) / 2 for pos in np.where(x_diff >= interpolation_size)[0])
        x = np.append(x, half_x)

        # while loop trap protection
        while_counter += 1
        if while_counter > 10:
            raise ValueError('Wrong interpolation size triggered while-loop escape.')


def fit_model(y_data, x_data, M0, M1, fit_params, f_statistic_params):
    """
    Fits the model M0 and M1 to the given x and y data. Particular execution is adjusted via the fit and f_statistic parameter dictionaries.
    First, x and y values are prepared and then fitted using the specified method. Then, parameter estimates, fold changes, and errors are calculated.
    Finally, the f-values are computed and corresponding p-values are obtained. All information about this process is returned.

    Parameters
    ----------
    y_data : pd.Series
        Series that contains the ratio y values.
    x_data : array-like
        An array-like object containing the drug concentrations in log space.
    M0 : MeanModel object
        An MeanModel instance from curve_curator.models.
    M1 : LogisticModel object
        An LogisticModel instance from curve_curator.models.
    fit_params : dict
        Parameter dictionary which adjusts the specific fitting procedures.
        It must contain at least the following key-value pairs:
            type : {OLS, MLE}
            speed : {fast, standard, extensive}
            control_fold_change : {True, False}
        The following key-value pairs are optional:
            weights : array-like with length equal to x_data and y_data
            interpolation : {True, False}
            x_interpolated : array-like, if interpolation is True
    f_statistic_params : dict
        Parameter dictionary which adjusts the specific fitting procedures.
         It must contain at least the following key-value pairs:
            optimized_dofs : {True, False}
            scale : float
            loc : float
        The following key-value pairs are optional:
            dfn : float
            dfd : float

    Returns
    -------
    *p_opt, fold_change, r2, M1.noise, *p_err, intercept, M0_noise, rmse, M1_likelihood, M0_likelihood, f_statistic, p_values
    """
    # Makes sure there is no carryover in the for-loop
    M1.reset()

    # Define x & y data for the fit
    y_data = y_data.values
    x_control = -np.inf
    y_control = 1
    weights = fit_params.get('weights')

    # Mask implausible y values and apply to x, y, weights
    finite_values = np.isfinite(y_data)
    if not all(finite_values):
        y_data = y_data[finite_values]
        x_data = x_data[finite_values]
        if weights is not None:
            weights = np.append(weights[0], weights[1:][finite_values])

    # Put x & y values together
    x = np.append(x_control, x_data)
    y = np.append(y_control, y_data)
    n = len(x)

    # ignore curve if there are to little number of data points and don't fit
    if n <= 4:
        return 19 * (np.nan,)

    # Interpolation helper points if wanted by the user. These are only applied during the fitting. Evaluation is purely based on the data.
    if fit_params.get('interpolation', False):
        f_linear = interpolate.interp1d(x_data, y_data, kind='linear')
        x_linear = fit_params['x_interpolated']
        # Mask terminal missing values
        if not all(finite_values):
            x_linear = x_linear[(x_linear >= x_data.min()) & (x_linear <= x_data.max())]
        # Add the values to the fit variables
        x_fit = np.append(x, x_linear)
        y_fit = np.append(y, f_linear(x_linear))
        weights = None
    else:
        x_fit = x
        y_fit = y

    # Fit the null model
    M0.set_initial_guess(intercept=np.mean(y_data), noise=np.std(y))
    if fit_params['type'] == 'OLS':
        M0.fit_ols(x_fit, y_fit)
    elif fit_params['type'] == 'MLE':
        M0.fit_mle(x_fit, y_fit)
    else:
        raise ValueError(f"Fit strategy type=\"{fit_params['type']}\" is not implemented.")
    intercept = M0.get_all_parameters()['intercept']
    rmse = M0.calculate_rmse(x_fit, y_fit)

    # Fit the unrestricted model with ordinary least squares (ols)
    if fit_params['type'] == 'OLS':
        if fit_params['speed'] == 'fast':
            M1.find_best_guess_ols(x_fit, y_fit, noise=M0.noise, weights=weights)
            M1.fit_ols(x_fit, y_fit, weights=weights)
        elif fit_params['speed'] == 'standard':
            M1.efficiently_fit_ols(x_fit, y_fit, noise=M0.noise, weights=weights)
        elif fit_params['speed'] == 'exhaustive':
            M1.extensively_fit_guesses_mle(x_fit, y_fit, noise=M0.noise)
        elif fit_params['speed'] == 'basinhopping':
            M1.find_best_guess_ols(x_fit, y_fit, noise=M0.noise, weights=weights)
            M1.basinhopping_ols(x_fit, y_fit, weights=weights)
        else:
            raise ValueError(f"Fit strategy speed=\"{fit_params['speed']}\" is not implemented for OLS.")

    # Fit the unrestricted model with maximum likelihood estimation (mle)
    elif fit_params['type'] == 'MLE':
        if fit_params['speed'] == 'fast':
            M1.find_best_guess_mle(x_fit, y_fit, noise=M0.noise)
            M1.fit_mle(x_fit, y_fit, weights=weights)
        elif fit_params['speed'] == 'standard':
            M1.efficiently_fit_mle(x_fit, y_fit, noise=M0.noise)
        elif fit_params['speed'] == 'extensive':
            M1.extensively_fit_guesses_mle(x_fit, y_fit, noise=M0.noise)
        else:
            raise ValueError(f"Fit strategy speed=\"{fit_params['speed']}\" is not implemented for MLE.")
    else:
        raise ValueError(f"Fit strategy type=\"{fit_params['type']}\" is not implemented.")

    # Get extra parameter
    M1.calculate_parameter_error(x, y)
    r2 = M1.calculate_r2(x, y)
    fold_change = M1.calculate_fold_change(x[1:], to_control=fit_params['control_fold_change'])
    auc = M1.get_auc(x)

    # Calculate f-statistic
    m0_sse = M0.calculate_sum_squared_residuals(x, y) + 1e-16
    m1_sse = M1.calculate_sum_squared_residuals(x, y) + 1e-16
    m1_k = M1.n_parameter()
    f_statistic = (m0_sse - m1_sse) / m1_sse * (n / m1_k)
    f_statistic = f_statistic if f_statistic >= 0.0 else 0.0

    # Calculate p-values for f-statistic based on parametric f-distribution
    dfn, dfd = M1.get_dofs(n, optimized=f_statistic_params.get('optimized_dofs'))
    dfn = f_statistic_params.get('dfn', dfn)
    dfd = f_statistic_params.get('dfd', dfd)
    p_values = stats.f.sf(f_statistic, dfn=dfn, dfd=dfd, scale=f_statistic_params['scale'], loc=f_statistic_params['loc'])

    # Get the parameters
    p_opt = M1.get_all_parameters().values()
    p_err = M1.get_params_error().values()

    return (*p_opt, fold_change, auc, r2, M1.noise, *p_err, intercept, M0.noise, rmse, M1.likelihood, M0.likelihood, f_statistic, p_values)


def add_logistic_model(df, ratio_cols, x_data, f_statistic_params, fit_params):
    """
    Fits a logistic model to the ratio columns. This is a wrapper function for the actual "fit_logistic_function(...)".

    Parameters
    ----------
    df : pd.DataFrame
        A data frame containing at least ratio_cols.
    ratio_cols : array_like
        A array-like object containing the column labels of the ratio data.
    x_data : array_like
        A array-like object containing the drug concentrations in log space.
    fit_params : dict
        Parameter dictionary which adjusts the specific fitting procedures.
        It must contain at least the following key-value pairs:
            type : {OLS, MLE}
            speed : {fast, standard, extensive}
            control_fold_change : {True, False}
        The following key-value pairs are optional:
            slope : 0 < float < 100
            front : float
            back : float
            weights : array-like with length equal to x_data and y_data
            interpolation : {True, False}
            x_interpolated : array-like, if interpolation is True
            max_iterations : int
    f_statistic_params : dict
        Parameter dictionary which adjusts the specific fitting procedures.
         It must contain at least the following key-value pairs:
            optimized_dofs : {True, False}
            scale : float
            loc : float
        The following key-value pairs are optional:
            dfn : float
            dfd : float

    Returns
    -------
    df : pd.DataFrame
        Output data frame with the fitted columns
    """
    # Define the logistic Model
    logistic_model = LogisticModel(slope=fit_params.get('slope'), front=fit_params.get('front'), back=fit_params.get('back'),
                                   max_iterations=fit_params.get('max_iterations'))
    logistic_model.set_boundaries(x_data)

    # Define the null model
    null_model = MeanModel(max_iterations=fit_params.get('max_iterations'))
    null_model.set_boundaries()

    # Fit the ratio data  to the logistic function
    fits = df[ratio_cols].progress_apply(fit_model,
                                         x_data=x_data,
                                         M0=null_model,
                                         M1=logistic_model,
                                         fit_params=fit_params,
                                         f_statistic_params=f_statistic_params,
                                         axis=1)

    # Typecast - apply output to DataFrame with the following column names
    fit_cols = ['pEC50', 'Curve Slope', 'Curve Front', 'Curve Back', 'Curve Fold Change', 'Curve AUC', 'Curve R2', 'Curve Noise',
                'pEC50 Error', 'Curve Slope Error', 'Curve Front Error', 'Curve Back Error',
                'Null Model', 'Null Noise', 'Null RMSE', 'Curve Likelihood', 'Curve Null Likelihood', 'Curve F_Value', 'Curve P_Value']
    df[fit_cols] = pd.DataFrame(data=fits.tolist(), columns=fit_cols, index=df.index)
    df['Curve Log P_Value'] = -np.log10(df['Curve P_Value'])

    # Only keep the likelihood for MLE estimation
    if fit_params['type'] != 'MLE':
        df.drop(columns=['Curve Noise', 'Null Noise', 'Curve Likelihood', 'Curve Null Likelihood'], inplace=True)
    return df


def run_pipeline(df, config, decoy_mode=False):
    """
    main function. Do the analysis based on the config file. Parallelize analysis with n cores.
    """
    # Load parameters from toml file
    experiments = np.array(config['Experiment']['experiments'])
    control_experiments = np.array(config['Experiment']['control_experiment'])
    drug_concs = np.array(config['Experiment']['doses'])
    drug_scale = float(config['Experiment']['dose_scale'])
    control_mask = (drug_concs != 0.0)
    drug_log_concs = tool.build_drug_log_concentrations(drug_concs[control_mask], drug_scale)

    # build the new column names based on experiment numbers
    cols_raw = tool.build_col_names('Raw {}', experiments)
    col_raw_control = tool.build_col_names('Raw {}', control_experiments)
    cols_normal = tool.build_col_names('Normalized {}', experiments)
    col_normal_control = tool.build_col_names('Normalized {}', control_experiments)
    cols_ratio = tool.build_col_names('Ratio {}', experiments)
    col_ratio_control = tool.build_col_names('Ratio {}', control_experiments)

    # Setup the curve fit with default values unless specified in the toml file
    proc_params = config['Processing']
    fit_params = config['Curve Fit']
    f_statistic_params = config['F Statistic']

    # Keep only rows with maximal n missing values excluding controls. Report filter effect to the user.
    k_rows_0 = len(df)
    df = filter_nans(df, cols_raw[control_mask], proc_params['max_missing'])
    k_rows_1 = len(df)
    if not decoy_mode:
        ui.message(f" * {k_rows_0 - k_rows_1} curves were removed because of >{proc_params['max_missing']} missing value(s).", end='\n')
        if k_rows_1 == 0:
            ui.error(f" * There is no curve left in the input data.", end='\n')
            exit()

    # Imputation of missing values if requested
    if proc_params['imputation'] and not decoy_mode:
        imputation_value = get_imputation_value(df, col_raw_control, pct=proc_params['imputation_pct'])
        df = impute_nans(df, cols_raw, imputation_value, proc_params['max_missing'])
        ui.message(f' * The following imputation value was used to fill NaNs: {round(imputation_value, 2)}', end='\n')

    # Normalize the data if requested
    if proc_params['normalization'] and not decoy_mode:
        df, norm_factors = normalize_values(df, cols_raw, cols_normal)
        out_path = config['Paths'].get('normalization_file', '')
        if out_path:
            norm_factors.to_csv(out_path, sep='\t', header=False, float_format='%.4f')
        ui.message(' * The following normalization factors were applied:', end='\n')
        ui.message('   {}'.format(norm_factors.round(2).to_dict()))

        # Calculate the ratios based on normalized values
        df = add_ratios(df, cols_normal, cols_ratio, col_normal_control)

    # Else calculate the ratios based on raw values
    else:
        df = add_ratios(df, cols_raw, cols_ratio, col_raw_control)

    # Filter rows where no ratios could be calculated because all controls were missing. Report filter effect to the user.
    k_rows_0 = len(df)
    df = filter_nans(df, col_ratio_control, max_missing=len(col_ratio_control)-1)
    k_rows_1 = len(df)
    if not decoy_mode:
        ui.message(f" * {k_rows_0 - k_rows_1} curves were removed because they had no valid control value(s).", end='\n')
        if k_rows_1 == 0:
            ui.error(f" * There is no curve left in the input data.", end='\n')
            exit()

    # If multiple controls are provided, estimate the noise level in the controls alone
    if len(col_raw_control) > 1:
        df['Control Ratio Std'] = df[col_ratio_control].std(axis=1)

    # Absolute signal quality is the raw intensity ot the control(s)
    df['Signal Quality'] = np.log2(df[col_raw_control].mean(axis=1))

    # Clip ratios if clipping is specified by the user
    if proc_params['ratio_range'] is not None:
        lower, upper = proc_params['ratio_range']
        clipped_value = ((df[cols_ratio] < lower) | (df[cols_ratio] > upper)).sum().sum()
        df[cols_ratio] = df[cols_ratio].clip(lower=lower, upper=upper)
        ui.message(f" * {clipped_value} ratios were clipped into the range [{lower}, {upper}].", end='\n')

    # Warn the user if negative values were detected or clip the values if requested
    negative_count = (df[cols_ratio] < 0).sum().sum()
    if negative_count > 0:
        ui.warning(f" * {negative_count} negative ratios were detected in the data matrix. CurveCurator expects the y response ratios to be >= 0." +
                   f"Models will be fit but will never go < 0. Please consider using the 'clip_ratios' parameter to confine the ratios into a defined range. e.g. (0, inf).", end='\n')

    # Sort concentrations and observations from low to high dose
    sorted_doses = np.argsort(drug_log_concs)
    drug_log_concs_sorted = drug_log_concs[sorted_doses]
    cols_ratio_sorted = tool.build_col_names('Ratio {}', experiments[control_mask][sorted_doses])
    if fit_params['weights'] is not None:
        fit_params['weights'] = np.append(fit_params['weights'][~control_mask], fit_params['weights'][control_mask][sorted_doses])

    if fit_params['interpolation']:
        fit_params['x_interpolated'] = build_interpolation_points(drug_log_concs_sorted)
        ui.message(' * Fit will use interpolation X values:', end='\n')
        ui.message('   {}'.format(list(map(lambda v: round(v, 2), fit_params['x_interpolated']))))

    # Fit the logistic model using multiple cores and optional processing parameters
    n_cores = config['Processing']['available_cores']
    data_type = 'decoy' if decoy_mode else 'curves'
    ui.message(f" * Fitting {data_type} parameters by {fit_params['speed']} {fit_params['type']} with {n_cores} cores:")
    df = tool.parallelize_dataframe(df, n_cores, add_logistic_model, ratio_cols=cols_ratio_sorted, x_data=drug_log_concs_sorted,
                                    f_statistic_params=f_statistic_params, fit_params=fit_params)
    ui.message(f' * Fitting {data_type} parameters done !')

    # Warn user if fixed parameter were used
    if any([fit_params.get('front'), fit_params.get('back'), fit_params.get('slope')]):
        ui.warning(' * The use of fixed model parameters is experimental and requires adjusted dfn and dfd parameter for correct p-value estimation !')
    return df
