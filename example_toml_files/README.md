
Each dataset comes with a parameter file in TOML (https://toml.io - v1.0.0) format. This file contains all necessary information for each experiment / raw input as well as optional parameters so that users can adjust the pipeline specifically to an experiment. The toml syntax primarily consists of `key = value` pairs, `[section names]`, and `#` (for comments). Example toml files, including extensive comments, are available. Common problems with the parameter file usually concern false formatting. Make sure strings have complete quotation marks. Lists are homogeneous in type, meaning that float and integers cannot be mixed, and that all elements are correctly separated by a comma. You don't need to specify all parameters all the time. Only specify parameters that differ from default behavior or are obligatory for the pipeline.

CurveCurator toml files have up to 7 `[sections]`. Obligatory ***`keys`*** are indicated with a bold and cursive font.
- `['Meta']` contains sample-specific information such as _**`id`**_, _**`description`**_, _**`condition`**_, _**`treatment_time`**_.

- `['Experiment']` contains information about the experimental design. CurveCurator has a generic file parser (default). For proteomics experiments, it can parse common search engine results directly. If this is wanted, those parameters are obligatory, too.    

	- ***`experiments`*** (array) is an id list containing the 1..N experiments names that are expected to be found in the data input file. Each name must be unique and exist in the data file. However, there can be more names in the data file than specified in the experiment array in the toml file. CurveCurator will only consider experiment columns specified here.
	- ***`doses`*** (array) contain the drug doses in a consistent order to the experiments array in float notation. The control channel(s) get a concentration of 0.0. The dose array gets scaled by the dose_scale parameter internally. This makes it possible to conveniently insert concentrations, e.g. in the nanomolar range.
	- ***`dose_scale`*** defines the unit prefix, and this gets multiplied with the doses array. For example, nano- is '1e-9', and micro- is '1e-6'. It's important to write it as a string in experimental notation or as float.
	- ***`dose_unit`*** the base unit of the doses and dose_scale. "M" for Molar is commonly used.
	- ***`control_experiment`*** specifies the experiment name(s) that contains the control (= 0.0 dose(s)). If there is a single control, specify the name. If multiple controls exist in the data, specify an array of controls, e.g. [1, 2, 3] for three control replicates with names 1, 2, and 3. Importantly, the names must match the id names in the experiments array.
  	- `measurement_type` (proteomic data) can be 'LFQ', 'TMT', 'DIA', 'OTHER'.
	- `data_type` (proteomic data) can be 'PEPTIDE', 'PROTEIN', 'OTHER'.
	- `search_engine` (proteomic data) can be 'MAXQUANT', 'DIANN', 'PD', 'MSFRAGGER', 'OTHER'.
	- `search_engine_version` (proteomic data) specifies the used version.

- `['Paths']` contains all path information that is relevant for the IO of the pipeline. Please note that all paths are relative to the toml file, which is currently executed. As a best practice, we recommend storing everything in one folder next to each other, resulting in the most simple relative paths possible. The paths are provided as follows: `path = './<path>/<file_name>.<extension>'`. The only path that you always need to specify is the input_file containing the raw data. All other paths serve to optionally rename the files or put them to another location than the default location next to the toml file.
	- ***`input_file`*** relative path to the raw data file.
	- `curves_file` relative path to the output file containing all fits and statistics.
	- `decoys_file` relative path to the output file containing all fits and statistics from the decoys. Only available in FDR mode.
	- `fdr_file` relative path to the output file containing the fdr estimate of the chosen relevance boundary. Only available in FDR mode.
	- `normalization_file` if data is normalized, the normalization factors can be stored in this file.  Only available if ['Processing'] normalization = true.
	- `mad_file` if MAD (median absolute deviation) analysis is performed to detect problematic experiments, the results are stored in this file. Only available in MAD mode.
	- `dashboard` will create an interactive bokeh plot for data exploration and analysis.

- `['Processing']` contains all optional parameters that are related to data (pre-) processing.
	- `available_cores` number of cores for parallelized fitting. The default value is 1 core.
	- `imputation` toggle if missing values should be imputed by a constant low value. This makes it very relevant for proteomic data, where missing values correlate with low abundance. If NANs are missing at random, don't set the toggle to true. The default is no imputation (=false).
	- `imputation_pct` is the specified percentile used for imputation. The default value is 0.005 (=0.5% raw intensity percentile of the control distribution).
    - `max_missing` number of maximally tolerated missing values per curve excluding the control(s). If an experiment has more than an accepted number of NANs, it is removed from the analysis. The default behavior is retaining all curves.
    - `max_imputation` number of maximally tolerated imputed values per curve excluding the controls(s). If an experiment has more than an accepted number of imputed values, it is removed from the analysis. The default behavior is retaining all curves.
    - `normalization` toggle if data should be globally normalized between experiments by a log-normal median-centric approach. This is important for proteomics data. The default behavior is not normalizing (=false).
    - `ratio_range` (array) specifies the lower and upper boundary of the ratio value range. If a value is outside the specified range, it will be clipped to the boundary. By default no ratio boundaries are applied, but the dose-response model cannot fit to negative values.

- `['Curve Fit']` contains all optional parameters that are related to the curve fitting procedure. For default behavior, nothing needs to be specified here.
	- `weights` (array) containing weights for the OLS fit, which increases the importance of some data points relative to others. The order is the same as in experiments and doses. A higher number means more importance. As a consequence, the curve will fit more closely to this data point. Default, each data point has the same importance (= all 1).
	- `type` specifies the fitting type, which is `'OLS'` for ordinary leased square fitting or `'MLE'` for maximum likelihood estimation. In our experience, both methods have similar overall performance. MLE is normally 10x slower but yields additional noise estimates and log-likelihoods for the model. The default is OLS.
	- `speed` specifies the fitting speed. Possible parameters include `'fast'`, `'standard'`, `'exhaustive'`, and `'basinhopping'`. Default is standard, which is a balance between conversion rate to the global minimum (best possible curve) and processing time and is the default fitting procedure. Fast is 3x faster than standard but finds less often the best possible curve. Exhaustive uses a big number of pre-defined start points to increase the chance of reaching the best curve but is >30x slower than standard. Basin hopping is a global minimum search algorithm that can overcome local minima by random perturbations and is >600 times slower than standard.
	- `max_iterations` specified the maximum number of iterations during the minimization process. Lower numbers can increase the overall speed.
	- `slope` can fix the curve slope of the fit to a pre-defined value. e.g., 1.0. All curves will have this slope value, and it is not present in the fitting procedure. Please note that fixing the value to a constant reduces the number of model parameters but must be based on good reasoning. This also has implications for the F-statistic and should be accounted for in the dfd and dfn calculations.
	- `front` can fix the curve front of the fit to a pre-defined value. e.g., 1.0. All curves will have this front value, and it is not present in the fitting procedure. Please note that fixing the value to a constant reduces the number of model parameters but must be based on good reasoning. This also has implications for the F-statistic and should be accounted for in the dfd and dfn calculations.
	- `back` can fix the curve back of the fit to a pre-defined value. e.g., 10.0. All curves will have this back value, and it is not present in the fitting procedure. Please note that fixing the value to a constant reduces the number of model parameters but must be based on good reasoning. This also has implications for the F-statistic and should be accounted for in the dfd and dfn calculations.
	- `control_fold_change` If true, CurveCurator will make the fold-change calculations relative to the control ratio. By default, the fold change is calculated between the minimal and maximal used dose.
	- `interpolation` If true, CurveCurator will generate interpolation points in-between data points during the fitting procedure. This makes the fitting procedure more robust against overfitting the data at the cost of not fitting the actual data points in the best possible way and thus reduces p-values slightly. Also, this can slow down the fitting step. By default, there is no interpolation.

- `['F Statistic']` contains all optional parameters that are related to the f-statistic, p-value calculation, and significance thresholding. Default values are optimized for the unconstrained 4-parameter sigmoidal curve.
	- ***`alpha`*** the significance threshold limit. This is the maximal p-value a curve can have.
	- ***`fc_lim`*** the fold change threshold limit. This is the minimal log2 fold change a curve can have (x-axis volcano plot). To convert it to a ratio threshold equivalent, you can transform it like this: ratio_lim = 2^(+-)fc_lim.
 	- `pEC50_filter` the range of valid pEC50 values. Relevant curves outside this pEC50 range will not be classified as down or up in the result files. This additional filter will be also be applied for decoy-curves in the "filtered FDR" estimation.
	- `loc`  location offset of the F-distribution.
	- `scale` scaling parameter of the F-distribution
	- `dfn` degrees of freedom of F-nominator (~ number of model parameters).
	- `dfd` degrees of freedom of F-denominator (~ number of free datapoints).
	- `optimized_dofs` Indicate if the optimized parametric degrees of freedom should be used. The default is True. If False, the standard degrees of freedom calculation for linear models will be used.
	- `quality_min` is the minimal quality that a curve needs to have to be trustworthy. This can be relevant for proteomics data with a low number of data points. As variance anticorrelates with intensity and signal/noise, it can be an efficient filter. The default is no filtering is applied.  
	- `mtc_method` specifies the technique for multiple testing correction. The default is no classical multiple testing, but the SAM inspired false positive rate reduction. To estimate the FDR of this SAM-like setting, you can activate the --fdr option when starting CurveCurator, which is the recommended way of analyzing dose-dependent data. Still, CurveCurator supports classical multiple-testing correction. The available correction methods come from [statsmodels.stats.multitest.multipletests](https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html#statsmodels-stats-multitest-multipletests "Permalink to this heading"). Please have a look there and use the same nomenclature.
	- `not_rmse_limit` specifies the root-mean-squared error threshold for the not classification. The default value is 0.1.
	- `not_p_limit` specifies an additional maximum p-value threshold for the not classification. The default uses no additional p-value filter.
	- `decoy_ratio` specifies the target decoy ratio. More decoys will improve FDR estimation but cost more analysis time.

- `['Dashboard']` contains all optional parameters to adjust the bokeh dashboard.
	- `backend`  Defines different bokeh backends to visualize the data. The default is "webgl", which facilitates fast rendering in the browser using the GPU. When saving plots during data exploration, they will be exported as non-editable .png-files. The backend can be changed to "svg" which allows the export of editable .svg-files. However, it can be a very slow experience in the browser, especially if there are more than 5k curves in the dataset. To get the default bokeh behavior, set it to "canvas" for HTML5 rendering.
