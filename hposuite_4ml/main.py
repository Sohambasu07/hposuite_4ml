"""HPOSuite class for regression tasks.
"""
from __future__ import annotations

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import logging
from typing import TYPE_CHECKING
import yaml
import pickle

from hpo_suite.hpo_glue.glu import GLUE
from hpo_suite.hpo_glue.obj_func import ObjectiveFunction
from hpo_suite.models.lgbm import LGBM
from hpo_suite.models.mlp import MLP
from hpo_suite.models.xgb import XGB
from hpo_suite.models.rf import RF

from hpo_suite.hpo_glue.glu import Experiment, Run
from hpo_suite.hpo_glue.problem import Problem, ProblemStatement

from hpo_suite.optimizers.random_search import RandomSearch         # noqa
from hpo_suite.optimizers.smac import (
    SMAC_Optimizer,                                                     # noqa
    SMAC_BO,                                                            # noqa
    SMAC_Hyperband                                                      # noqa
)
from hpo_suite.optimizers.dehb import DEHB_Optimizer                # noqa  
from hpo_suite.optimizers.optuna import OptunaOptimizer             # noqa
from hpo_suite.optimizers.synetune import (
    SyneTuneOptimizer,                                                  # noqa
    SyneTune_BOHB,                                                      # noqa
    SyneTune_KDE,                                                       # noqa
)

if TYPE_CHECKING:
    from hpo_suite.hpo_glue.optimizer import Optimizer
    from pathlib import Path

GLOBAL_SEED = 0

models_list = {
    "lgbm": LGBM,
    "xgb": XGB,
    "rf": RF,
    "mlp": MLP,
}

logger = logging.getLogger(__name__)

class HPOSuite:

    def __init__(
        self,
        dataset_name: str,
        num_seeds: int = 1,
        metric: str = "r2",
        model_names: str | list [str] | None = None,
        kfold: bool = False,
        warm_start_kfold: bool = False
    ) -> None:
        self.dataset_name = dataset_name
        self.num_seeds = num_seeds
        self.metric = metric
        self.model_names = model_names
        self._best_model_name = None
        self.kfold = kfold
        self.warm_start_kfold = warm_start_kfold


    def data_setup(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ) -> list[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            random_state=GLOBAL_SEED,
            test_size=0.2,
        )
        return X_train, X_val, y_train, y_val

    def glu_setup(
        self,
        train_dataset: tuple[np.ndarray, np.ndarray],
        test_dataset: tuple[np.ndarray, np.ndarray],
        exp_config: str,
        model_dir: Path,
        budget_type: str = "n_trials",
        budget: int = 100,
        num_workers: int = 1,
        exp_name: str = "exp",
    )-> Experiment:

        problems = []

        # Getting valid ProblemStatements

        with open(exp_config, "r") as f:
            config = yaml.safe_load(f)

        for model_name in models_list:
            model = models_list[model_name]()
            for instance in config["optimizer_instances"]:
                problem_statement = ProblemStatement(
                    objective_function = ObjectiveFunction(
                        name = f"{model_name}_{self.dataset_name}",
                        dataset_name = self.dataset_name,
                        train_dataset = train_dataset,
                        test_dataset = test_dataset,
                        model = model,
                        metric = self.metric,
                        warm_start = self.warm_start_kfold,
                        ws_model_name = model_dir / f"best_model_{self.dataset_name}.pkl"
                    ),
                    optimizer = eval(config["optimizer_instances"][instance]["optimizer"]),
                    hyperparameters = config["optimizer_instances"][instance]["hyperparameters"],
                )
                problem = Problem(
                    problem_statement = problem_statement,
                    objectives = self.metric,    #TODO: Add multiobjective support
                    minimize = False,        #TODO: Change in case of multiobjective
                    fidelities = model.fidelity   #TODO: Explicitly defined in case of dataset fidelity
                )
                if self.sanity_checks(
                    optimizer = problem_statement.optimizer,
                    problem = problem,
                    budget_type = budget_type,
                    budget = budget
                ):
                    problems.append(problem)
                
        if len(problems) == 0:
            raise ValueError("No valid problems could be created")
        
        logger.info(f"Problems created: {len(problems)}")

        # Creating a Run
                    
        run = Run(
            budget_type = budget_type,
            budget = budget,
            num_seeds = self.num_seeds,
            problems = problems
        )

        # Creating an Experiment

        exp = Experiment(
            name = exp_name,
            runs = [run],
            n_workers = num_workers
        )

        return exp

    def run_hposuite(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        exp_config: str,
        exp_name: str,
        save_dir: str,
        model_dir: str,
        budget_type: str = "n_trials",
        budget: int = 100,
        num_workers: int = 1,
    ) -> HPOSuite:
        X_train, X_val, y_train, y_val = self.data_setup(X, y)

        if self.model_names is None:
            model_names = list(models_list.keys())
        elif isinstance(self.model_names, str):
            model_names = [self.model_names]

        for model_name in model_names:
            assert model_name in models_list, f"Model {model_name} not found"

        
        # Creating GLUE Experiment

        exp = self.glu_setup(
            train_dataset = (X_train, y_train),
            test_dataset = (X_val, y_val),
            exp_config = exp_config,
            model_dir = model_dir,
            budget_type = budget_type,
            budget = budget,
            num_workers = num_workers,
            exp_name = exp_name
        )

        # Running the Experiment

        exp_dir, best_score, self.best_model_name = GLUE.experiment(
            experiment = exp,
            save_dir = save_dir,
            model_dir = model_dir
        )
        logger.info(f"Experiment results saved at {exp_dir}")
        logger.info(f"Best model: {self.best_model_name}")
        logger.info(f"Best score: {best_score}")

        return self

    def sanity_checks(
        self,
        optimizer: Optimizer,
        problem: Problem,
        budget_type: str,
        budget: int
    ) -> bool:

        # Check if model has fidelity space for MF optimization
        if optimizer.supports_multifidelity and problem.fidelities is None:
                raise ValueError(
                    "Invalid Problem Statement! "
                    f"{optimizer.name} is a multifidelity optimizer but "
                    f"model {problem.problem_statement.objective_function.model.name} "
                    "has no fidelity space defined!"
                )
        
        # Do not allow fidelity_budget using non-MF optimizer
        if not optimizer.supports_multifidelity and budget_type == "fidelity_budget":
            raise ValueError(
                "Invalid Problem Statement! "
                f"{optimizer.name} does not support multifidelity optimization "
                "but budget type is set to fidelity_budget"
            )
        
        # Check if budget is less than the maximum fidelity value for fidelity_budget
        if optimizer.supports_multifidelity and budget_type == "fidelity_budget":
            if budget < problem.problem_statement.objective_function._model.fidelity_space[-1]:
                raise ValueError(
                    "Invalid Problem Statement! "
                    "fidelity_budget type but budget is less than the "
                    "maximum fidelity value for the model "
                    f"{problem.problem_statement.objective_function._model.name}"
                )
        


        return True


    def predict(
        self,
        model_dir: Path,
        X: pd.DataFrame,
    ) -> np.ndarray:
        X = X.to_numpy()
        if self.best_model_name is None:
            raise ValueError("Model not fitted")

        with open(model_dir / f"best_model_{self.dataset_name}.pkl", "rb") as f:
            best_model = pickle.load(f)

        return best_model.predict(X)  # type: ignore
