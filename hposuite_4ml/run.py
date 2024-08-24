"""An example run file which loads in a dataset from its files
and logs the R^2 score on the test set.

In the example data you are given access to the y_test, however
in the test dataset we will provide later, you will not have access
to this and you will need to output your predictions for X_test
to a file, which we will grade using github classrooms!
"""
from __future__ import annotations

from pathlib import Path
from sklearn.metrics import r2_score
import numpy as np
from hposuite_4ml.data import Dataset
from hposuite_4ml.main import HPOSuite
import argparse
import os

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
    fold: int,
    kfold: bool,
    warm_start_kfold: bool,
    pred_path: Path,
    budget_type: str = "n_trials",
    budget: int = 100,
    num_workers: int = 1,
    num_seeds: int = 1,
    metric: str = "r2",
    models: str | list[str] | None = None,
    exp_name: str = "myexp",
    exp_config: str = "exp_configs.yaml",
    datadir: Path = DATADIR,
    savedir: Path = SAVEDIR,
    model_dir: Path = MODEL_DIR
):

    logger.info("Fitting HPOSuite")

    if not kfold:
        warm_start_kfold = False
    else:
        logger.warning(
            "Running on all k-folds."
            " Existing saved models will skew results!"
        )

    exp_config = CONFIGS_DIR/exp_config

    match kfold:
        case False:
            if task == "all":
                tasks = list(task_ids.keys())
                tasks.remove("exam_dataset")
                for i, task in enumerate(tasks):
                    logger.info(f"RUNNING ON TASK {task}, task no. {i+1} out of {len(tasks)}")
                    logger.info(f"RUNNING ON FOLD {fold}")
                    hposuite = HPOSuite(
                        dataset_name=task,
                        num_seeds=num_seeds,
                        metric=metric,
                        model_names=models,
                        kfold=kfold,
                        warm_start_kfold=warm_start_kfold
                    )
                    dataset = Dataset.load(datadir=datadir, task=task, fold=fold)
                    _ = run_pipeline(
                        hposuite=hposuite,
                        task=task,
                        dataset=dataset,
                        exp_config=exp_config,
                        budget_type=budget_type,
                        budget=budget,
                        num_workers=num_workers,
                        exp_name=exp_name,
                        savedir=savedir,
                        model_dir=model_dir,
                        pred_path=pred_path
                    )
                
            else:
                logger.info(f"RUNNING ON TASK {task}")
                logger.info(f"RUNNING ON FOLD {fold}")
                hposuite = HPOSuite(
                    dataset_name=task,
                    num_seeds=num_seeds,
                    metric=metric,
                    model_names=models,
                    kfold=kfold,
                    warm_start_kfold=warm_start_kfold
                )
                dataset = Dataset.load(datadir=datadir, task=task, fold=fold)
                _ = run_pipeline(
                    hposuite=hposuite,
                    task=task,
                    dataset=dataset,
                    exp_config=exp_config,
                    budget_type=budget_type,
                    budget=budget,
                    num_workers=num_workers,
                    exp_name=exp_name,
                    savedir=savedir,
                    model_dir=model_dir,
                    pred_path=pred_path
                )

        case True:
            folds = [int(f) for f in os.listdir(datadir / task_ids[task]) if f.isnumeric()]
            folds.sort()
            scores = []
            hposuite = HPOSuite(
                dataset_name=task,
                num_seeds=num_seeds,
                metric=metric,
                model_names=models,
                kfold=kfold,
                warm_start_kfold=warm_start_kfold
            )
            for fold in folds:
                logger.info(f"RUNNING ON FOLD {fold}")
                dataset = Dataset.load(datadir=datadir, task=task, fold=fold)
                score = run_pipeline(
                    hposuite=hposuite,
                    task=task,
                    dataset=dataset,
                    exp_config=exp_config,
                    budget_type=budget_type,
                    budget=budget,
                    num_workers=num_workers,
                    exp_name=exp_name,
                    savedir=savedir,
                    model_dir=model_dir,
                    pred_path=pred_path
                )
                scores.append(score)
            logger.info(f"Mean R^2 score on the test sets of all folds: {np.mean(scores)}")
                

def run_pipeline(
        hposuite: HPOSuite,
        task: str,
        dataset: Dataset,
        exp_config: Path,
        budget_type: str,
        budget: int,
        num_workers: int,
        exp_name: str,
        savedir: Path,
        model_dir: Path,
        pred_path: Path
)-> float:
    hposuite.run_hposuite(
        X = dataset.X_train,
        y = dataset.y_train,
        exp_config = exp_config,
        budget_type = budget_type,
        budget = budget,
        num_workers = num_workers,
        exp_name = exp_name,
        save_dir = savedir,
        model_dir = model_dir
    )

    logger.info("HPOSuite pipeline complete")

    logger.info("Predicting on test set")
    test_preds: np.ndarray = hposuite.predict(
        model_dir = model_dir,
        X = dataset.X_test
    )
        # Write the predictions of X_test to disk
    # This will be used by github classrooms to get a performance
    # on the test set.
    if task != "exam_dataset":
        logger.info("Writing predictions to disk")
        with pred_path.open("wb") as f:
            np.save(f, test_preds)
    else:
        logger.info("Writing predictions for exam dataset to disk")
        with open(DATADIR / "exam_dataset" / "1" / "predictions.npy", "wb") as f:
            np.save(f, test_preds)

    r2_test = -np.inf

    if dataset.y_test is not None:
        r2_test = r2_score(dataset.y_test, test_preds)
        logger.info(f"R^2 on test set: {r2_test}")
    else:
        # This is the setting for the exam dataset, you will not have access to y_test
        logger.info(f"No test set for task '{task}'")
    
    print("=======================================================\n")

    return r2_test




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The name of the task to run on.",
        choices=["y_prop", "bike_sharing", "brazilian_houses", "exam_dataset", "all"]
    )
    parser.add_argument(
        "--pred-path",
        type=Path,
        default=Path("predictions.npy"),
        help=(
            "The path to save the predictions to."
            " By default this will just save to './predictions.npy'."
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
        "--kfold",
        action="store_true",
        help=(
            "Whether to run on all k-folds of tha dataset."
            "NOTE: This is not kfold cross validation!"
            " If this is set, the fold argument will be ignored."
        )
    )
    parser.add_argument(
        "--warm_start_kfold", "-ws",
        action="store_true",
        help=(
            "Whether to warm start the models trained on previous fold"
            "when running on multiple dataset folds. Only applicable if kfold is set."
        )
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=1,
        help=(
            "The number of seeds to use for the experiment."
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
        "--quiet",
        action="store_false",
        help="Whether to log only warnings and errors."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=None,
        help="The models to use for the experiment"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="r2",
        help="The metric to optimize"
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="exp_configs.yaml",
        help="The path to the experiment configuration file"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=SAVEDIR,
        help="The path to save the results"
    )
    parser.add_argument(
        "--budget_type",
        type=str,
        default="n_trials",
        help="The type of budget"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=10,
        help="The budget"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="myexp",
        help="The name of the experiment"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=MODEL_DIR,
        help="The path to the saved models"
    )


    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logger.info(
        f"Running task {args.task}"
        f"\n{args}"
    )

    main(
        task=args.task,
        fold=args.fold,
        kfold=args.kfold,
        warm_start_kfold=args.warm_start_kfold,
        pred_path=args.pred_path,
        num_seeds=args.num_seeds,
        metric=args.metric,
        models=args.models,
        budget_type=args.budget_type,
        budget=args.budget,
        num_workers=args.num_workers,
        exp_name=args.exp_name,
        exp_config=args.exp_config,
        datadir=args.datadir,
        savedir=args.save_dir,
        model_dir=args.model_dir
    )
