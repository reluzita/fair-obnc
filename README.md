# Fair-OBNC: Fair Ordering-Based Label Noise Correction

## Running an experiment

Run the script `run_noiseinjection.py` for experiments using standard ML datasets, or `run.py` for experiments using fairness benchmark datasets that do not require the noise injection step.

Used datasets are stored to a `data` folder.

During the experiments, the corrected labels are saved to a `correction` folder, and the model predictions are saved to a `predictions` folder. The results are stored through MLFlow, in a `mlruns` folder.

The conducted experiments were executed using the `run_standard_datasets.sh` and `run_fairness_datasets.sh` script files.  

## Result analysis

Examples of how to analyze the obtained results according to the used methodology for the systematic analysis of label noise correction methods in achieving ML fairness (including the visualizations that were used in our work), can be found in the `result_analysis.ipynb` notebook.

In this notebook we load the results from csv files, instead of directly from MLFlow, which can be generated using the `maintenance.ipynb` notebook.

## Implementation notes

- `evaluation.py` contains the evaluation functions, mostly implementations of fairness measures and functions for logging results to MLFlow;
- `format_data.py` contains the functions for loading data from OpenML and formatting it for the experiments;
- `label_correction.py` contains the implementation of the used and proposed label noise correction methods;
- `noise_injection.py` contains the functions for manipulating labels to inject different types of label noise.


