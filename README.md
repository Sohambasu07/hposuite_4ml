# HPOSuite for Machine Learning

`hposuite_4ml` provides a one-click AutoML pipeline generation. It Interfaces HPO Optimizers from different 
libraries with Machine Learning models and datasets for benchmarking and comparative analysis. Other features like 
Meta Learning, Feature Importance and Plotting are also included.

> **_NOTE:_**  This package was built for the AutoML course at the University of Freiburg, SS2024.


## Installation

To install the repository, first create an environment of your choice and activate it. 

**Virtual Environment**

```bash
python3 -m venv hs4ml_env
source hs4ml_env/bin/activate
```

**Conda Environment**

Can also use `conda`, left to individual preference.

```bash
conda create -n hs4ml_env python=3.10.12
conda activate hs4ml_env
```

Then install the repository by running the following command:

```bash
pip install -e .
```

You can test that the installation was successful by running the following command:
```bash
python -c "import hposuite_4ml"
```

## Python Version
Python version 3.10.12 or high is recommended

## Code
We provide the following:

* `download-openml.py`: A script to download the dataset from openml given a `--task`, corresponding to an OpenML Task ID. We wil provide those suggested as training datasets, prior to us releasing the test dataset.

* `run.py`: A script that loads in a downloaded dataset, executes the HPOSuite AutoML pipeline using the config and CLI arguments
and then generates predictions for `X_test`, saving those predictions to a file.

## Data

### Sample datasets:

* [y_prop_4_1 (361092)](https://www.openml.org/search?type=task&id=361092&collections.id=299&sort=runs)
* [Bike_Sharing_Demand (361099)](https://www.openml.org/search?type=task&id=361099&collections.id=299&sort=runs)
* [Brazilian Houses (361098)](https://www.openml.org/search?type=task&id=361098&collections.id=299&sort=runs)

The OpenML sample datasets can be downloaded using:
```bash
python download-openml.py --task <task_id>
```

This will by default, download the data to the `/data` folder with the following structure.
The fold numbers, `1, ..., n` are **outer folds**, meaning you can treat each one of them as
a seperate dataset for training and validation. You can use the `--fold` argument to specify which fold you would like.

```bash
./data
├── 361092
│   ├── 1
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
│   ├── 2
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
│   ├── 3
    ...
├── 361098
│   ├── 1
│   │   ├── X_test.parquet
│   │   ├── X_train.parquet
│   │   ├── y_test.parquet
│   │   └── y_train.parquet
    ...
```

## Running an initial test
This will train an AutoML system using Optuna-TPE Sample and XGBoost Regressor and generate predictions for `X_test`:
```bash
python hposuite_4ml/run.py --task bike_sharing --num_seeds 1 --output-path preds-s1-bikesh.npy --models xgb --exp_config dev_config.yaml
```

## Run instructions


### Command line arguments:

`--task`: Name of the dataset/task

`--fold`: fold to run on

`--num_seeds`: Number of seeds to use for the run

`--datadir`: directory where the datasets are stored

`--exp_config`: yaml config file name inside the ./configs directory. The yaml config file should have a set of optimizer name and optimizer hyper-hyperparameters

`--budget_type`: The type of budget to use. The ones implemented are "n_trials", "time_budget" and "fidelity_budget".\
"n_trials": Number of functions evaluations for non-multifidelity optimzers like SMAC_BO and Optuna_TPE. Doesn't work for multifidelity optimizers like Hyperband and DEHB.\
"fidelity_budget": Total fidelity budget to use for Multifidelity Optimizers. Only works for MF optimizers
"time_budget": Wallclock time limit for any HP Optimizer. Works for all times.

`--exp_name`: Name of the Experiment. Detailed Optimizer run results are stored in a parquet file inside the Experiment directory.\
Directory Structure: `<exp_name>\_\<datetime>/Run\_\<budget_type>\_\<budget>/\<task>\/<Optimizer_Name>/\<Seed>/report\_\<task>\_\<Optimzier_name>\_\<Optimzier_Hyper-hyperparameter>\_\<Model_name>.parquet

`--model_dir`: Directory where the saved models are stored\
Saved models are stored in the format: `best_model_<dataset_name>.pkl`

### Best model and best model configs:

The bash command provided above will yield a trained model and a hyperparameter configuration for the `--task` argument (dataset name) entered.

Hyperparameter configs are stored in:\
`model_dir\best_model_<task>_config.json`

For example: This repo contains the final test configs in:
`saved_models/best_model_exam_dataset_config.json`\
and best model pickle file in:
`saved_models/best_model_exam_dataset.pkl`

### Assumptions

The code works on the following assumptions, based on the initial repository provided:

1. The `task` argument is the name of the dataset which is mapped to a dataset in the `task_ids` dict in `run.py`
2. The train set of the entered `task` dataset is a parquet file which is loaded using `hposuite_4ml/data.py`
3. The output predictions are stored in `./predictions.npy` which is ignored by git as set in the `.gitignore` file.
4. Exception to point `3` is when the dataset is named `exam_dataset`. In that case the predictions are saved under `./data/exam_dataset/1/predictions.npy`.
5. The `exp_name` directory is placed under the `./results` folder


### Example configs:

The file `./configs/exp_configs.yaml` consists an example of what a yaml config file for optimizers should look like

The file `./configs/final_test_config.yaml` consists of the optimizer config for the final run on the test dataset.

The file `./final_test_predictions.npy` consists of the final predictions on the exam_dataset's test set.


### Run results

The final parquet file consists of a lot of information about the run including the id of each config, seed, optimizer name, optimizer hyper-hyperparameters, model name, configuration, objective name, objective value, time and fidelity costs, fidelity type, etc.

### Plotting

The `./src/hpo_suite/hpo_glue/utils.py` file consists of the script for plotting from the parquet files. The only parameter needed really is the `--exp_dir` which under the `./results` folder.


