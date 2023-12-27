# quality_control.py
# Quality control of the input data to identify potential systematic biases.
#
# Florian P. Bayer - 2024
#

# Imports:
import numpy as np

from . import toolbox as tool
from . import user_interface as ui
from .models import LogisticModel

from tqdm.autonotebook import tqdm
tqdm.pandas()


def calc_residual(row, drug_c, y_col_names, fit_col_names):
    """
    calculates the ratio difference between the prediction y_hat and the observation y.
    """
    model = LogisticModel()
    model.set_fitted_params(row[fit_col_names])
    return model.residuals(x=drug_c, y=row[y_col_names])


def calc_deviations(df, drug_c, y_col_names, fit_col_names):
    """
    Wrapper function to allow for parallelization of calc_residual.
    """
    residuals = df.progress_apply(calc_residual, drug_c=drug_c, y_col_names=y_col_names, fit_col_names=fit_col_names, axis=1)
    return residuals


def mad_analysis(df, config):
    """
    main function. Do the analysis based on the config file. Parallelize analysis with n cores.
    """
    # Perform MAD analysis if out path is given.
    out_path = config['Paths'].get('mad_file')
    if not out_path:
        return None
    ui.message(' * MAD-Analysis:')

    # Pars the toml data
    experiments = np.array(config['Experiment']['experiments'])
    cols_ratio = tool.build_col_names('Ratio {}', experiments)
    drug_concs = np.array(config['Experiment']['doses'])
    drug_scale = float(config['Experiment'].get('dose_scale'))
    drug_log_concs = tool.build_drug_log_concentrations(drug_concs, drug_scale, dmso_offset=1e5)
    n_cores = int(config['Processing'].get('available_cores'))
    cols_fit = ['pEC50', 'Curve Slope', 'Curve Front', 'Curve Back']

    # Calculate the deviations with multiple cores and then the mad based on this
    cols = np.concatenate([cols_ratio, cols_fit])
    dev = tool.parallelize_dataframe(df[cols], n_cores, calc_deviations, drug_c=drug_log_concs, y_col_names=cols_ratio, fit_col_names=cols_fit)
    mad = dev.abs().median()
    mad.to_csv(out_path, sep='\t', header=False, float_format='%.4f')
    ui.message(' * MAD-Analysis found the following median absolute deviations:', end='\n')
    ui.message('   {}'.format(mad.round(2).to_dict()))
