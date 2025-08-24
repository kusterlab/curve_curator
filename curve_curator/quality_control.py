# quality_control.py
# Quality control of the input data to identify potential systematic biases.
#
# Florian P. Bayer - 2025
#

# Imports:
import numpy as np

from . import toolbox as tool
from . import user_interface as ui
from .models import LogisticModel


def calc_residual(row, drug_c, y_col_names, fit_col_names):
    """
    calculates the ratio difference between the prediction y_hat and the observation y.
    """
    model = LogisticModel()
    model.set_fitted_params(row[fit_col_names])
    return (row.name, *model.residuals(x=drug_c, y=row[y_col_names]))


def mad_analysis(df, config):
    """
    main function. Do the analysis based on the config file. Parallelize analysis with n cores.
    """
    # Perform MAD analysis if out path is given.
    out_path = config['Paths'].get('mad_file')
    if not out_path:
        return None
    ui.message(' * MAD-Analysis:')

    # Pars the toml data.
    experiments = np.array(config['Experiment']['experiments'])
    cols_ratio = tool.build_col_names('Ratio {}', experiments)
    cols_res = tool.build_col_names('MAD {}', experiments)
    drug_concs = np.array(config['Experiment']['doses'])
    drug_scale = float(config['Experiment'].get('dose_scale'))
    drug_log_concs = tool.build_drug_log_concentrations(drug_concs, drug_scale, dmso_offset=1e5)
    n_cores = int(config['Processing'].get('available_cores'))
    cols_fit = ['pEC50', 'Curve Slope', 'Curve Front', 'Curve Back']

    # Calculate the residuals with multiple cores and then the mad based on this for each column in cols_ratio.
    calc_residual_kwargs = {'drug_c':drug_log_concs, 'y_col_names':cols_ratio, 'fit_col_names':cols_fit}
    cols = np.concatenate([cols_ratio, cols_fit])
    residuals = tool.parallelize_dataframe(df[cols], func=calc_residual, return_cols=cols_res, func_kwargs=calc_residual_kwargs, n_cores=n_cores)
    mad = residuals.abs().median()
    mad.to_csv(out_path, sep='\t', header=False, float_format='%.4f')
    ui.message(' * MAD-Analysis found the following median absolute deviations:', end='\n')
    ui.message('   {}'.format(list(map(lambda i: f'{str(i[0])}: {round(float(i[1]), 2)}', mad.items()))))
