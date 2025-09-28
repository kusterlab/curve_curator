# Different example data sets

In the folders above, we have reprocessed real data that should serve as an example of how to use CurveCurator.

The data sets originate from the following papers:
- decryptM (https://doi.org/10.1126/science.ade3925)
- Kinobeads (https://doi.org/10.1126/science.aan4368)
- Viability_CTRP (https://doi.org/10.1016/j.cell.2013.08.003)
- Viability_Sarcoma (https://doi.org/10.1038/s44320-023-00004-7) → With examples for different replicate analysis strategies

# General comments

CurveCurator always expects tab ("\t") separated files as input. If you use a different file encoding it will raise an error when reading the data.

CurveCurator has different parser modes that enable convenient data import. In the following, we will describe the different parser modes and how you correctly match your data file to your parameter.toml file. Many combinations of measurement_type, data_type, and search_engine are possible. We will only highlight a few combinations but many more are possible. In case you get a `NotImplementedError`, please let us know. We will implement the specific parser mode for you then.

# Generic Parser: (OTHER, OTHER, OTHER)

This is the "swiss knife"-mode that will always work. To activate it, set in the toml file:

```
['Experiment']
experiments = ['A', 'B 2', ... , 'N']
data_type = 'OTHER'
measurement_type = 'OTHER'
search_engine = 'OTHER'
```

The data structure should look like this:

| Name    | Raw A  | Raw B 2 | ... | Raw N  |
| ------- | ------ | ------- | --- | ------ |
| Name_1  | 1010.0 | 1025.0  | ... |  880.0 |
| Name_2  | 1210.0 |  999.9  | ... | 1050.0 |
| Name_3  |  950.0 |   80.3  | ... |    2.0 |

- The "Name" column serves as a unique key for a dose-response curve. If multiple same names exist, CurveCurator will aggregate the duplicates.
- The experiment names are indicated after "Raw <exp_name>" and separated with a whitespace. The name can also contain multiple whitespaces.
- Each experiment name in the experiment list in the toml file must exist in the data file. But columns that are not in the experiment file will be ignored for the analysis.
- The raw values can be any float or int and also contain missing values.

# Generic Protein Parser: (PROTEIN, OTHER, OTHER)

This adds some protein functionality to the generic importer.

```
['Experiment']
experiments = ['A', 'B 2', ... , 'N']
data_type = 'PROTEIN'
measurement_type = 'OTHER'
search_engine = 'OTHER'
```

The data structure should look like this:

| Genes | Proteins        | Raw A  | Raw B 2 | ... | Raw N  | Score | Peptides |
| ----- | --------------- | ------ | ------- | --- | ------ | ----- | -------- |
| GENEA | P00001          | 1010.0 | 1025.0  | ... |  880.0 | 800   | 30       |
| GENEB | Q00001          | 1210.0 |  999.9  | ... | 1050.0 | 120   |  2       |
| GENEC | P00002;P00003   |  950.0 |   80.3  | ... |    2.0 | 501   | 12       |

- "Gene" and "Proteins" columns are a combined unique column instead of the previous introduced Name column.
- The experiment names are indicated after "Raw <exp_name>" and separated with a whitespace. The name can also contain multiple whitespaces.
- Each experiment name in the experiment list in the toml file must exist in the data file. But columns that are not in the experiment file will be ignored for the analysis.
- The raw values can be any float or int and also contain missing values.
- "Score" and "Peptides" columns are optional columns that can be plotted / shown in the dashboard and curves.txt file. However, they are not necessary, and can be left out. CurveCurator will warn you that these columns are missing.

# Generic Peptide Parser: (PEPTIDE, OTHER, OTHER)

This adds some peptide functionality to the generic importer.

```
['Experiment']
experiments = ['A', 'B 2', ... , 'N']
data_type = 'PEPTIDE'
measurement_type = 'OTHER'
search_engine = 'OTHER'
```

The data structure should look like this:

| Modified sequence | Genes | Proteins        | Raw A  | Raw B 2 | ... | Raw N  | Score |
| ----------------- | ----- | --------------- | ------ | ------- | --- | ------ | ----- |
| ABCDEFG           | GENEA | P00001          | 1010.0 | 1025.0  | ... |  880.0 | 800   |
| HIJKLMNOPQ        | GENEB | Q00001          | 1210.0 |  999.9  | ... | 1050.0 | 120   |
| KLKS(ph)K(ac)LK   | GENEC | P00002;P00003   |  950.0 |   80.3  | ... |    2.0 | 501   |

- The "Modified sequence" column serves as a unique key for a dose-response curve. If multiple rows have the same modified sequence, CurveCurator will aggregate the duplicates.
- The experiment names are indicated after "Raw <exp_name>" and separated with a whitespace. The name can also contain multiple whitespaces.
- Each experiment name in the experiment list in the toml file must exist in the data file. But columns that are not in the experiment file will be ignored for the analysis.
- The raw values can be any float or int and also contain missing values.
- The "Genes", "Proteins", and "Score" columns are optional columns that can be plotted / shown in the dashboard and curves.txt file. However, they are not necessary, and can be left out. CurveCurator will warn you that these columns are missing.

# Custom parser for proteomics search engines

- For proteomics data, use the search engine of your choice and specify it simply in the toml file. Please search each dose-dependent experiment (one condition e.g. a single drug) separately. If you search multiple experiments together, you must split the result files before subjecting it to CurveCurator. Otherwise CurveCurator can get confused and aggregates the same protein or peptide from different experiments.
- You must specify if it is peptide or protein data.
- Experience has schon that when you simply name your experiments 1..N in the search engine, it has the highest success rate to correctly set up the TOML file. But everything is possible.
- For TMT this is already done by most search engines.

- For MAXQUANT, use the protein.txt file for protein-based analysis and the evidence.txt for peptide-based analysis.
- For DIANN, it outputs raw file names as columns. Please rename manually to Raw 1..N.
- For PD, the order of the files is important. PD normally labels the output experiments with F1..N. These numbers will be parsed by the CurveCurator. Please make sure that toml file has the same N to dose correspondences.
- For MSFRAGGER, name your TMT channels or LFQ experiments Raw_1...N manually. The peptide-based analysis expects the (combined_)ion.tsv file. The protein-based analysis expects the (combined_)protein.tsv file.
