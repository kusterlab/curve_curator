For Viability data, create a .txt file (tab separated) containing a "Name" column used as a sample identifier and Raw value, which can be intensities or ratios.

For proteomics data, use the search engine of your choice and specify it in the toml file.
Also, you must specify if it is peptide or protein data. Please search each dose-dependent experiment (one condition e.g. a single drug) separately.
Name your experiments 1..N in the search engine. For TMT this is already done by most search engines.
For MAXQUANT, use the protein.txt file for protein-based analysis and the evidence.txt for peptide-based analysis.
For DIANN, it outputs raw file names as columns. Please rename manually to Raw 1..N.
For PD, the order of the files is important. PD normally labels the output experiments with F1..N. These numbers will be parsed by the CurveCurator. Please make sure that toml file has the same N to dose correspondences.
For MSFRAGGER, name your TMT channels or LFQ experiments Raw_1...N. The peptide-based analysis expects the (combined_)ion.tsv file. The protein-based analysis expects the (combined_)protein.tsv file.







# Different example data sets

The data sets originate from the following papers:
- decryptM (https://doi.org/10.1126/science.ade3925)
- Kinobeads (https://doi.org/10.1126/science.aan4368)
- Viability_CTRP (https://doi.org/10.1016/j.cell.2013.08.003)
- Viability_Sarcoma (https://doi.org/10.1038/s44320-023-00004-7) → With examples for different replicate analysis strategies
