"""
To perform metalearning on the best model configurations
OR
To just predict on the test set using the best model
"""

from __future__ import annotations

from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from hposuite_4ml.data import Dataset
import argparse
import pickle
import json
from hposuite_4ml.hpo_suite.models.lgbm import LGBM         # noqa
from hposuite_4ml.hpo_suite.models.xgb import XGB           # noqa
from hposuite_4ml.hpo_suite.models.rf import RF             # noqa
from hposuite_4ml.hpo_suite.models.mlp import MLP           # noqa

import logging

logger = logging.getLogger(__name__)

FILE = Path(__file__).absolute().resolve()
DATADIR = FILE.parent / "data"
SAVEDIR = FILE.parent / "results"
CONFIGS_DIR = FILE.parent / "configs"
MODEL_DIR = FILE.parent / "saved_models"

task_ids = {
    "y_prop": "361092",
    "bike_sharing": "361099",
    "brazilian_houses": "361098",
    "exam_dataset": "exam_dataset"
}

def main(
    task: str,
    metalearned_task: str,
    fold: int,
    pred_path: Path,
    perform: str = "predict",
    datadir: Path = DATADIR,
    model_dir: Path = MODEL_DIR
):

    logger.info("Predicting using HPOSuite")

    
    logger.info(f"RUNNING ON TASK {task}")
    logger.info(f"RUNNING ON FOLD {fold}")

    # Loading the dataset
    dataset = Dataset.load(datadir=datadir, task=task, fold=fold)

    if perform == "metalearn":

        # Loading model configs
        with open(model_dir / f"best_model_{metalearned_task}_config.json", "r") as f:
            model_config = json.load(f)

        config = {}
        model_name = ""

        for model_names in model_config:
            match model_names:
                case "LightGBM_Regressor":
                    model_name = "LGBM"
                case "XGBoost_Regressor":
                    model_name = "XGB"
                case "RandomForest_Regressor":
                    model_name = "RF"
                case "MLP_Regressor":
                    model_name = "MLP"
                case _:
                    raise ValueError(f"Model {model_names} not supported")
            for opt in model_config[model_names]:
                for dataset_name in model_config[model_names][opt]:
                    for config_list in model_config[model_names][opt][dataset_name]:
                        if config_list == "config":
                            config = model_config[model_names][opt][dataset_name][config_list]
                        else:
                            continue

        model = eval(model_name)()
        seed = config["random_state"]
        config.pop("random_state")

        logger.info("Fitting the model")
        X_train, X_val, y_train, y_val = train_test_split(
            dataset.X_train, dataset.y_train, test_size=0.2, random_state=0
        )
        model.fit(
            seed = seed,
            config = config,
            X_train = X_train.to_numpy(),
            y_train = y_train.to_numpy()
        )
        model.predict(X_val.to_numpy())
        logger.info("Evaluating the model on validation set")
        r2_val = r2_score(y_val, model.predict(X_val.to_numpy()))
        logger.info(f"R^2 on validation set: {r2_val}")

    elif perform == "predict":
        with open(model_dir / f"best_model_{task}.pkl", "rb") as f:
            model = pickle.load(f)

    logger.info("Predicting on the test set using the best model")
    test_preds = model.predict(dataset.X_test.to_numpy())
    r2_test = None
    if task != "exam_dataset":
        logger.info(f"R^2 on test set: {r2_test}")
        r2_test = r2_score(dataset.y_test, test_preds)
    logger.info("Writing predictions to disk")
    with pred_path.open("wb") as f:
        np.save(f, test_preds)
    
    print("=======================================================\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--load_task",
        type=str,
        required=True,
        help="The name of the task to run on.",
        choices=["y_prop", "bike_sharing", "brazilian_houses", "exam_dataset"]
    )
    parser.add_argument(
        "--metalearned_task",
        type=str,
        required=False,
        help="The name of the task whose metalearned model to load.",
        choices=["y_prop", "bike_sharing", "brazilian_houses", "exam_dataset"]
    )
    parser.add_argument(
        "--pred-path",
        type=Path,
        default=Path("predpy_predictions.npy"),
        help=(
            "The path to save the predictions to."
            " By default this will just save to './predictions.npy'."
            " Only works for the exam dataset."
        )
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help=(
            "The fold to run on."
            " You are free to also evaluate on other folds for your own analysis."
            " For the test dataset we will only provide a single fold, fold 1."
        )
    )
    parser.add_argument(
        "--datadir",
        type=Path,
        default=DATADIR,
        help=(
            "The directory where the datasets are stored."
            " You should be able to mostly leave this as the default."
        )
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=MODEL_DIR,
        help="The path to the saved models"
    )
    parser.add_argument(
        "--perform",
        type=str,
        default="predict",
        help="The operation to perform. Either predict or metalearn",
        choices=["predict", "metalearn"]
    )


    args = parser.parse_args()

    if args.perform == "metalearn" and args.metalearned_task is None:
        raise ValueError("Please provide the task whose metalearned model to load")

    main(
        task=args.load_task,
        metalearned_task=args.metalearned_task,
        fold=args.fold,
        pred_path=args.pred_path,
        datadir=args.datadir,
        model_dir=args.model_dir,
        perform=args.perform
    )    
