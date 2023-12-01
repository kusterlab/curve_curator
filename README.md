![CurveCurator](logo.png)


[![DOI](https://img.shields.io/badge/Paper-10.1038%2Fs41467--023--43696--z-be2635?logo=Paper&link=https%3A%2F%2Fdoi.org%2F10.1038%2Fs41467-023-43696-z)](https://doi.org/10.1038/s41467-023-43696-z)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8399823.svg)](https://doi.org/10.5281/zenodo.8399823)
[![PyPI version](https://badge.fury.io/py/curve-curator.svg)](https://badge.fury.io/py/curve-curator)


# CurveCurator

CurveCurator is an open-source analysis platform for dose-dependent data sets. It fits a classical 4-parameter equation to estimate effect potency, effect size, and the statistical significance of the observed response. 2D-thresholding efficiently reduces false positives in high-throughput experiments and separates relevant from irrelevant or insignificant hits in an automated and unbiased manner. An interactive dashboard allows users to quickly explore data locally.

For more information, we refer to the paper. Especially the supplementary notes contain many explanations and tips and tricks for your data analysis strategy, which is dependent on the specific data set that you obtained.

## Installation:

#### 1. Install the virtual environment manager anaconda to install CurveCurator and its dependencies safely.
If you have anaconda already installed on your computer, you can move to step 2. If you still need anaconda, please go to the website (https://www.anaconda.com/) and download the newest version for your operating system. With the anaconda installation, you get an "Anaconda Prompt". This shell will be needed to install and execute the program later. If you are more advanced, you can use other shells too.

#### 2. Install a new environment for CurveCurator in the shell.
Open the "Anaconda Prompt" program. Installation of the environment is only required once. Type the following command into the shell: 
```sh
(base)$ conda create -n CurveCuratorEnv pip
```
This will create a new environment with the name CurveCuratorEnv. In this environment, it will install the "pip" software. It will ask you to confirm the installation of some packages for pip. If you have an environment with the same name already created or you prefer a different name, you must either delete it or create an environment with a different name. Please remember the name of the environment. For more information, see the anaconda documentation. 

#### 3. Activate the CurveCuratorEnv environment
Activation of the curve_curator environment is always required each time you open a new shell and must be done before you run the pipeline (see section "Run pipeline").
```sh
(base)$ conda activate CurveCuratorEnv
...
(CurveCuratorEnv)$ 
```
Successful activation is confirmed by seeing the name of the current environment in the braces before the $. 

#### 4. Install CurveCurator and its dependencies in the CurveCuratorEnv
We have registred CurveCurator in PyPi (https://pypi.org/project/curve-curator/). This allows fast installation of the latest stable version using the following pip command. Make sure you are in the correct environment.
```sh
(CurveCuratorEnv)$ pip install curve-curator
```
Verify installation by seeing that the program exists. If everything was done correctly, you will see the help output of CurveCurator (as shown below) and you are done with the installation.
```sh
(CurveCuratorEnv)$ CurveCurator -h
...
usage: CurveCurator [-h] [-b] [-f] [-m] [-r [RANDOM]] <PATH>

CurveCurator

positional arguments:
  <PATH>                Relative path to the config.toml or batch.txt file to run the pipeline.

options:
  -h, --help            show this help message and exit
  -b, --batch           Run a batch process with a file containing all the parameter file paths.
  -f, --fdr             Estimate FDR based on target decoy approach. Estimating the FDR will double the run time.
  -m, --mad             Perform the medium absolute deviation (MAD) analysis to detect outliers
  -r [RANDOM], --random [RANDOM] Run the pipeline with <N> random values for H0 simulation.
```
If you see instead the message: " 'CurveCurator' is not recognized as an internal or external command, operable program or batch file" or any other error, then there was a problem during the installation. Also, double-check that you are in the correct environment.

If you want to update CurveCurator to the latest version after you have installed it already, redo the pip install of step 4.

## Preparation:

#### 1. Prepare the raw data


For Viability data, create a txt file (tab separated) containing a "Name" column used as a sample identifier and Raw value, which can be intensities or ratios.

For proteomics data, use the search engine of your choice and specify it in the toml file. 
Also, you must specify if it is peptide or protein data. Please search each dose-dependent experiment (one condition e.g. a single drug) separately. 
Name your experiments 1..N in the search engine. For TMT this is already done by most search engines.
For MAXQUANT, use the protein.txt file for protein-based analysis and the evidence.txt for peptide-based analysis. 
For DIANN, it outputs raw file names as columns. Please rename manually to Raw 1..N.
For PD, the order of the files is important. PD normally labels the output experiments with F1..N. These numbers will be parsed by the CurveCurator. Please make sure that toml file has the same N to dose correspondences. 
For MSFRAGGER, name your TMT channels or LFQ experiments Raw_1...N. The peptide-based analysis expects the (combined_)ion.tsv file. The protein-based analysis expects the (combined_)protein.tsv file.

#### 2. Fill out the parameter toml-file for each dataset
Each dataset comes with a parameter file in toml format. This file contains all necessary information for each experiment / raw input as well as optional parameters so that users can adjust the pipeline specifically to an experiment. The toml syntax primarily consists of `key = value` pairs, `[section names]`, and `#` (for comments). Example toml files, including extensive comments, are available. Common problems with the parameter file usually concern false formatting. Make sure strings have complete quotation marks. Lists are homogeneous in type, meaning that float and integers cannot be mixed, and that all elements are correctly separated by a comma. You don't need to specify all parameters all the time. Only specify parameters that differ from default behavior or are obligatory for the pipeline.

CurveCurator toml files have up to 7 `[sections]`. Obligatory ***`keys`*** are indicated with bold and cursive font.
- `['Meta']` contains sample-specific information such as _**`id`**_, _**`description`**_, _**`condition`**_, _**`treatment_time`**_.

- `['Experiment']` contains information about the experimental design. CurveCurator has a generic file parser (default). For proteomics experiments, it can parse common search engine results directly. If this is wanted, those parameters are obligatory, too.    

	- ***`experiments`*** (array) is an id list containing the 1..N experiments names that are expected to be found in the data input file. Each name must be unique and exist in the data file. However, there can be more names in the data file than specified in the experiment array in the toml file. CurveCurator will only consider experiment columns specified here.
	- ***`doses`*** (array) contain the drug doses in a consistent order to the experiments array in float notation. The control channel(s) get a concentration of 0.0. The dose array gets scaled by the dose_scale parameter internally. This makes it possible to conveniently insert concentrations, e.g. in the nanomolar range.
	- ***`dose_scale`*** defines the unit prefix, and this gets multiplied with the doses array. For example, nano- is '1e-9', and micro- is '1e-6'. It's important to write it as a string in experimental notation or as float. 
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
	- `max_missing` number of maximally tolerated missing values per curve excluding the control(s). If an experiment has more than an accepted number of NANs, it is removed from the analysis. If imputation is activated, then this will also limit the maximal number of allowed imputations per experiment. The default behavior is retaining all experiments.
	- `normalization` toggle if data should be globally normalized between experiments by a log-normal median-centric approach. This is important for proteomics data. The default behavior is not normalizing (=false).

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

- `['F Statistic']` contains all optional parameters that are related to the f-statistic, p-value calculation, and significance thresholding. Default values are optimized for the unconstrained 4-paramter sigmoidal curve. 
	- ***`alpha`*** the significance threshold limit. This is the maximal p-value a curve can have.
	- ***`fc_lim`*** the fold change threshold limit. This is the minimal log2 fold change a curve can have (x-axis volcano plot). To convert it to a ratio threshold equivalent, you can transform it like this: ratio_lim = 2^(+-)fc_lim.
	- `loc`  location offset of the F-distribution.
	- `scale` scaling parameter of the F-distribution
	- `dfn` degrees of freedom of F-nominator (~ number of model parameters).
	- `dfd` degrees of freedom of F-denominator (~ number of free datapoints).
	- `optimized_dofs` Indicate if the optimized parametric degrees of freedom should be used. Default is True. If False, the standard degrees of freedom calculation for linear models will be used.
	- `quality_min` is the minimal quality that a curve needs to have to be trustworthy. This can be relevant for proteomics data with a low number of data points. As variance anticorrelates with intensity and signal/noise, it can be an efficient filter. The default is no filtering is applied.  
	- `mtc_method` specifies the technique for multiple testing correction. Default is no classical multiple testing but the SAM inspired false positive rate reduction. To estimate the FDR of this SAM-like setting, you can activate the --fdr option when starting CurveCurator, which is the recommended way of analyzing dose-dependent data. Still CurveCurator supports classical multiple testing correction. The availible correction methods come from [statsmodels.stats.multitest.multipletests](https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html#statsmodels-stats-multitest-multipletests "Permalink to this heading"). Please have a look there and use the same nomenclature.
	- `not_rmse_limit` specifies the root-mean-squared error threshold for the not classification. Default value is 0.1.
	- `not_p_limit` specifies an additional maximum p-value threshold for the not classification. Default uses no additional p-value filter.
	- `decoy_ratio` specifies the target decoy ratio. More decoys will improve FDR estimation but cost more analysis time.

- `['Dashboard']` contains all optional parameters to adjust the bokeh dashboard.
	- `backend`  Defines different bokeh backends to visualize the data. Default is "webgl", which facilitates fast rendering in the browser using the GPU. When saving plots during data exploration, they will be exported as non-editable .png-files. The backend can be changed to "svg" which allows the export of editable .svg-files. However, it can be a very slow experience in the browser, especially if there are more than 5k curves in the dataset. To get the default bokeh behavior, set it to "canvas" for HTML5 rendering.



## Run pipeline
There are multiple modes to execute CurveCurator:

#### Mode 1. Run the pipeline script for one dataset
The standard way of running the script is shown below. All necessary information is provided via the toml file. Make sure that you are in the correct environment, which is called 'CurveCuratorEnv' if you have followed the installation guide. 
```sh
(base)$ conda activate CurveCuratorEnv
(CurveCuratorEnv)$ CurveCurator <toml_path>
```
There are optional parameter to enable additional analysis steps. If the FDR option is activated, Curve Curator will generate decoys based on the data input and estimate the false discovery rate (FDR) for the user-given alpha and fold-change additionally. If the MAD option is activated, the noisy channel detection is performed additionally.
```sh
(CurveCuratorEnv)$ CurveCurator <toml_path> --fdr --mad
```

#### Mode 2. Run the pipeline script for many datasets as batch
The batch_file is just a txt file containing a list of toml file paths that will be processed consecutively. FDR and MAD parameters can be activated optionally.
```sh
(CurveCuratorEnv)$ CurveCurator --batch <batch_file> --fdr --mad
```

#### Mode 3. Run the pipeline with simulated data
If you apply non-standard settings and want to experiment with F-value distributions under the H0=true, you can simulate your own distributions. N indicates the number of curves you want to simulate. The generated data will be saved as the input file specified in the toml file. 
```sh
(CurveCuratorEnv)$ CurveCurator <toml_path> --random <N>
```

## Explore data with the interactive dashboard

CurveCurator provides the user with an interactive dashboard. Different functionalities are accessible depending on the specific dataset. All dashboards consist of a global plot on the left side (either volcano plot view or potency plot view), a dose-response curve area in the middle showing selected curves, data selection tools, and a data table giving more information about selected items. Each plot has a toolbar (upper right corner) that allows for data engagement, such as panning, tap or lasso selection, resetting, and saving. Thanks to the hover tool, a small information box appears, showing the name of the dot/curve. There are also a few keyboard shortcuts e.g., multi-selection and de-selection. For more in-depth information, please visit the bokeh documentation. Data from the HTML file cannot be deleted or altered. Refreshing the browser will revert all adjustments, filters, and selections to the default. If a particular representation is of interest, it can be exported as a figure via the save tool. On smaller laptop screens, it is possible that the canvas width exceeds the screen width. Unfortunately, bokeh cannot rescale the width automatically. There are two possibilities to deal with this situation: 1) You either accept it and scroll left and right to plots of interest; or 2) you zoom out until it matches your screen. A quick reload of the HTML page in the browser will remove the blurriness that may arise as a consequence of rescaling. 


The **Volcano plot** relates the Log2 - Curve Fold Change (CFC) vs. the Curve Significance (-log10 p-value) or Curve Relevance Score (CRS). Each dot is one dose-response curve, and the color indicates its potency. Negative CFC values indicate down-regulation and positive CFC values indicate up-regulation. Please remember that the CFC is normally defined as the log2 ratio between the lowest and highest concentration - not to the control - unless you actively change this via the toml parameter "control_fold_change=true". The x and y axis automatically scales to the data, showing the complete range of values in the dataset. The red line shows the decision boundary, which was constructed based on the toml file (alpha & fc_lim asymptotes). One can hide or show curves using the all/regulated/not-regulated toggle at the top. By clicking on a curve(s), the volcano plot shows that it was selected by blurring the non-selected dots. Simultaneously, the selected curve appears in the dose-response area and in the table. In the volcano plot view, there are additional buttons next to the view toggle that allow you to switch between p-values and relevance scores. The relevance score is based on the s0 SAM statistic and describes the statistical significance and the biological relevance in a single number. If specified in the toml file, other p-value adjustment techniques can be used instead of the relevance score. Here, the p-value and fold change cutoffs are two independent boundaries.

The **Potency plot** relates the Curve Fold Change (CFC) vs. the Curve Potency (pEC50). It can be accessed via the drop-down menu at the top. By default, only significant curves are shown since only those pEC50 values can be interpreted. Please, never interpret curve estimates from insignificant curves. Hovering, clicking, and other functionalities are identical to the Volcano plot.

The **Dose-Reponse Curve area** plots only selected curves and yields a quick overview of the raw data and the fitted curve. There is no other functionality. The curves can be exported.

The **Histogram area** displays a few extra values (when present in the input data), which can be helpful in interpreting specific curves. When a curve is selected, a red line indicates where in the distribution the selected curve is located. Again, hovering will show the name of the curve. The black dashed lines indicate the current selection thresholds, which are additionally applied to the data set (see below) to focus on a specific subset of curves.

The **Curve Selection area** helps to select and filter curves. Depending on the different dataset types, sliders filter for a subset of curves (pEC50, Score, Signal). Curves that are not within the selected range will become invisible. The thresholds are indicated in the histograms corresponding to the slider. Below the sliders, there are search fields for selecting curves by strings. In fact, these are regex-compatible search strings, allowing for complex querying of the data. For example, all peptide sequences containing the motive of a proline-driven serine-threonine kinase can be selected and visualized by searching the sequence against `[S|T]\(ph\)P`. Please have a look at Python regex notations for more details. Please also note that hidden/filtered curves cannot be selected via the search fields. Only curves that are displayed in the Volcano or Potency plot are selectable. If you are looking for a curve that is hidden for some reason, you need to remove the filters (sliders or regulations) first.

The **Table** provides more detailed information about each curve. By clicking on the table's headers, it can be sorted alpha-numerically. By clicking on a table row, only the specific row will be selected, and the rest will be de-selected. When holding the ctrl-key while clicking, only this row is de-selected, and the rest stays selected.


## FAQ:
Q: The regulation column in the curves files has categories up, down, not. However, many rows are not classified into these categories. Why? and How should I interpret this?

A: Not classified curves should be treated very carefully. In principle, there are two reasons why a curve regulation type could not be determined. First, the curve is too noisy. There is no apparent regulation, but the data points are so scattery that one should simply not interpret anything here. Second, the curve has low noise but also exhibits some sort of faint regulation; just the fold change was not big enough to render it relevant. However, it would be an overinterpretation to call these curves not regulated.

Q: How to deal with replicated doses in one experiment?

A: Curve curator can deal with replicated data. It is also possible to have only replicated controls and no replicates for the same doses. There are different possibilities for handling replicates in CurveCurator. 1) It's possible to get a single curve from replicated data where the replicated doses were aggregated to a single average point before the fitting. 2) It's possible to get a single curve with all replicated ratios being fitted simultaneously. In the dashboard, you are able to see all individual observations around the estimated curve. 3) It's possible to get an independent curve fit for each replicate experiment. Depending on the selected strategy 1-3, the data structure and the toml file need to be adapted accordingly. 

Q: What is the Relevance Score?

A: After fitting the curve model to the observed response, CurveCurator calculates an F-value and p-value for each regression curve. The user has then defined an alpha threshold (to control statistical significance) and a fold change threshold (to define biological relevance) that are both used to find high-quality curves in the dataset. The relevance score combines these two properties of significance and biological relevance into a single number for each curve. Consequently, the previous hyperbolic decision boundary in the classical volcano plot will be a single relevance threshold in the alternative volcano plot after the transformation. 
