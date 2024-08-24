# HPOSuite for Machine Learning

## Run instructions

### Python command to reproduce the test predictions:

```bash
python -m run --task "exam_dataset" --budget 100 --exp_config "final_test_config.yaml" --exp_name "final_test_optuna_lgbm" --num_seeds 5
```

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
2. The train set of the entered `task` dataset is a parquet file which is loaded using `./src/automl/data.py`
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


