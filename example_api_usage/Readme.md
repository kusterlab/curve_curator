# API

CurveCurator offers a nice API to fit and evaluated dose-response curves.

### Import CurveCurator as package: 

```python
import curve_curator
curve_curator.__version__
```

### Package Structure:

- models:

Contains the Mean and Logistic model that is used to fit data. See example notebooks for usage details.
```python
from curve_curator import models

m0 = models.MeanModel()
m1 = models.LogisticModel()
```


- quantification:

Contains functions to clean, filter, and manipulate data.
Please see the docstring for detailed information about the specific parameters.
```python
from curve_curator import quantification

df = get_some_data()
cols = ['Name 1', ...]

quantification.filter_nans(df, cols, max_missing)
quantification.get_imputation_value(df, col, pct=0.005)
quantification.impute_nans(df, raw_cols, imputation_value, max_imputations)
quantification.normalize_values(df, raw_cols, norm_cols, ref_col=None)
quantification.add_ratios(df, cols, ratio_cols, ref_cols)
```


- thresholding:

Contains functions to perform Relevance-Score analysis.
Please see the docstring for detailed information about the specific parameters.
```python
from curve_curator import thresholding

thresholding.get_s0(fc_lim, alpha, dfn, dfd, loc=0, scale=1, two_sided=False)
thresholding.get_fclim(s0, alpha, dfn, dfd, loc=0, scale=1, two_sided=False)
thresholding.map_fc_to_pvalue_cutoff(x, alpha, s0, dfn, dfd, loc=0, scale=1, two_sided=False)
thresholding.correct_pvalues(pvalues, alpha=0.01, method='fdr_bh')
thresholding.sam_correction(f_values, curve_fold_change, s0)
thresholding.calculate_qvalue(df, sort_cols, sort_ascendings, decoy_col, q_col_name='Curve q-value')
```