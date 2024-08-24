from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import os
import datetime
import time
from tqdm import tqdm
import logging
import pickle
import json

from hposuite_4ml.hpo_suite.hpo_glue.history import History
from hposuite_4ml.hpo_suite.hpo_glue.problem import Problem
from hposuite_4ml.hpo_suite.hpo_glue.state import State

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class Run:

    budget_type: str
    """The type of budget to use for the optimizer.
    Currently supported: ["n_trials", "time_budget", "fidelity_budget"]"""

    budget: int
    """The budget to run the optimizer for"""

    num_seeds: int | None
    """The number of seeds to use for the Runs in the Experiment"""

    problems: list[Problem]
    """The Problems inside the Run to optimize"""


    def __init__(
        self,
        budget_type,
        budget,
        problems,
        num_seeds
    ) -> None:
        
        self.budget_type = budget_type
        self.budget = budget
        self.problems = problems
        self.num_seeds = num_seeds
        if self.num_seeds is None:
            self.num_seeds = 1


class Experiment:

    name: str
    """The name of the Experiment"""

    runs: list[Run]
    """The list of Runs inside the Experiment"""

    n_workers: int | None
    """The number of workers to use for the Problem Runs"""


    def __init__(
        self,
        name,
        runs,
        n_workers = 1
    ) -> None:
        self.name = name
        self.runs = runs
        self.n_workers = n_workers


class GLUEReport:
    optimizer_name: str
    obj_func_name: str
    problem: Problem
    history: pd.DataFrame

    def __init__(
        self, 
        optimizer_name: str, 
        obj_func_name: str, 
        problem: Problem,
        history: pd.DataFrame,
    ) -> None:
        self.optimizer_name = optimizer_name
        self.obj_func_name = obj_func_name
        self.problem = problem
        self.history = history


class GLUE:
    root: Path = Path(os.getcwd())
    GLOBAL_SEED = 42

    def run(problem: Problem,
        run_dir: Path | str,
        model_dir: Path | str,
        budget_type: str,
        budget: int,
        seed: int | None = None,
        best_run_score: float = -np.inf,
        progress_bar: bool = False
    ) -> float:
        """Runs an optimizer on an objective function, returning a report."""

        # if isinstance(exp_dir, str):
        #     exp_dir = Path(exp_dir)

        # if not os.path.exists(exp_dir):
        #     exp_dir = GLUE.root / exp_dir
        #     os.makedirs(exp_dir)

        # run_dir = f"Run_{budget_type}_{budget}"
        # run_dir = exp_dir / run_dir

        budget_num = 0
        history = History()

        optimizer = problem.problem_statement.optimizer
        objective_function = problem.problem_statement.objective_function

        opt = optimizer(problem = problem, 
                        working_directory = (
                            GLUE.root / 
                            "Optimizers_cache" / 
                            f"{optimizer.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        ),
                        seed = seed,
                        **problem.problem_statement.hyperparameters
                        )
        

        if not problem.problem_statement.hyperparameters:
            optimizer_name = f"{optimizer.name}_"
        else:
            optimizer_name = (
                f"{optimizer.name}_"
                f"{list(problem.problem_statement.hyperparameters.keys())[0]}_"
                f"{list(problem.problem_statement.hyperparameters.values())[0]}"
            )

        opt_hps = optimizer_name.split(f"{optimizer.name}_")[1]

        max_bud = budget

        if budget_type != "n_trials" and budget_type != "time_budget" and budget_type != "fidelity_budget":
            raise ValueError(f"Budget type {budget_type} not supported!")

        logger.info(
            f"Running Problem: {problem.problem_statement.name},"
            f" Optimizer: {optimizer.name}_{problem.problem_statement.hyperparameters}" 
            f" Model: {problem.problem_statement.objective_function._model.name} budget: {budget}, "
            f"budget_type: {budget_type}, seed: {seed}\n"
            )
        
        unit = ''
        match budget_type:
            case "n_trials": 
                unit = "trial"
            case "fidelity_budget":
                unit = "fidelity"
            case "time_budget":
                unit = "s"

        if progress_bar:
            prog_bar = tqdm(total = budget, desc="Budget used", unit=unit)

        start_time = time.time()

        best_prob_score = -np.inf

        while budget_num < budget:

            query = opt.ask(config_id=str(budget_num))

            # Placing fidelity budget here to save unnecessary out of budget evaluations

            # NOTE: Fidelity Budget cannot be run with non MF optimizers

            if budget_type == "fidelity_budget":
                budget_num += query.fidelity
                if budget_num > budget:
                    if progress_bar:
                        prog_bar.update(budget - prog_bar.n)
                    logger.info("Budget exhausted!")
                    break

            #Model fitting time
            st = time.time()
            result = objective_function(seed=seed, query=query)
            et = time.time()
            fit_time = et - st


            if budget_type == "time_budget":
                budget_num += fit_time
                if budget_num > budget:
                    if progress_bar:
                        prog_bar.update(budget - prog_bar.n)
                    logger.info("Budget exhausted!")
                    break

            res_update = {
                "fit_time": fit_time,
                "fidelity_type": problem.problem_statement.objective_function._model.fidelity
            }
            result.result.update(res_update)

            score = result.result[problem.objectives]
            opt.tell(result)

            best_prob_score = max(best_prob_score, score)
            if progress_bar:
                prog_bar.set_description_str(desc=f"Best score of this problem: {best_prob_score}")


            # Updating the budget and checking if it is exhausted

            if budget_type == "n_trials":
                if progress_bar:
                    prog_bar.update(1)
                budget_num += 1

            elif budget_type == "time_budget":
                if progress_bar:
                    prog_bar.update(fit_time)

            elif budget_type == "fidelity_budget":
                if progress_bar:
                    prog_bar.update(query.fidelity)

            # if budget_num > budget:
            #     logger.info(f"Budget exhausted!")
            #     break 

            if score > best_run_score:
                best_run_score = score
                if progress_bar:
                    prog_bar.set_description_str(desc=f"New best score!: {best_run_score}")

                # Saving the best model
                with open(
                    model_dir / f"best_model_"
                    f"{problem.problem_statement.objective_function.dataset_name}.pkl", 
                    "wb"
                ) as f:
                    pickle.dump(problem.problem_statement.objective_function._model._model, f)

                # Saving the configs of the best model
                json_extra = {
                    problem.problem_statement.objective_function._model.name: {                             # Model name
                        f"{optimizer_name}": {                                                              # Optimizer name
                            problem.problem_statement.objective_function.dataset_name: {                    # Dataset name
                                "config": query.config.values,                                              # Config values
                                "fidelity": {
                                    "type": problem.problem_statement.objective_function._model.fidelity,   # Fidelity type
                                    "value": query.fidelity if query.fidelity is not None else None         # Fidelity value
                                },
                                "val_score": score,                                                         # Validation score
                                "fit_time": fit_time,                                                       # Model fitting time
                            }
                        }
                    }
                }
                with open(
                    model_dir / f"best_model_"
                    f"{problem.problem_statement.objective_function.dataset_name}_config.json", 
                    "w"
                ) as f:
                    json.dump(json_extra, f, indent=4)
                                    
            history.add(result)  
        
        end_time = time.time()

        if not progress_bar:
            logger.info(f"Best score of this problem: {best_prob_score}")

        logger.info(
            f"Time taken for Problem {problem.problem_statement.name}:"
            f" {end_time - start_time} seconds"
        )

        cols = (
            ["config_id", "fidelity"]
            + list(result.query.config.values.keys())
            + list(result.result.keys())
        )

        if "fit_time" not in cols:
            cols.append("fit_time")
        if "fidelity_type" not in cols:
            cols.append("fidelity_type")

        hist = history.df(cols)
        hist['max_budget'] = max_bud
        hist['minimize'] = problem.minimize
        hist['objectives'] = problem.objectives
        hist["budget_type"] = budget_type
        hist["budget"] = budget
        hist["seed"] = seed
        hist["runtime_cost"] = end_time - start_time
        hist["optimizer_name"] = optimizer_name
        hist['optimizer_hyperparameters'] = opt_hps
        hist["objective_function"] = objective_function.name

        history._save(
            report = hist,
            runsave_dir = run_dir,
            dataset_name = problem.problem_statement.objective_function.dataset_name,
            model_name = problem.problem_statement.objective_function._model.name,
            optimizer_name = optimizer.name,
            optimizer_hyperparameters = opt_hps,
            seed = seed
        )
        
        
        print(hist)
        # print(hist["Fidelity"].value_counts())

        return best_run_score
    
    
    def generate_seeds(
        num_seeds: int,
    ):
        """Generate a set of seeds using a Global Seed."""
        _rng = np.random.default_rng(GLUE.GLOBAL_SEED)
        seeds = _rng.integers(0, 2 ** 30 - 1, size=num_seeds)
        return seeds
    
    def experiment(
        experiment: Experiment,
        save_dir: Path | str,
        model_dir: Path | str,
        root_dir: Path | str = Path(os.getcwd())
    ):
        """Runs an experiment, returning a report."""

        best_score = -np.inf #NOTE: Only applicable for maximization (r2)
        best_model_name = None

        # Creating current Experiment directory
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        GLUE.root = root_dir
        save_dir = root_dir / save_dir
               
        exp_dir = f"Exp_{experiment.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir = save_dir / exp_dir

        # Creating the directory to save the models
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)

        if not os.path.exists(model_dir):
            model_dir = root_dir / model_dir
            os.makedirs(model_dir)

        # Running the experiment using GLUE.run()
        for i, run in enumerate(experiment.runs):
            logger.info(f"Executing Run: {i}")

            run_dir = f"Run_{run.budget_type}_{run.budget}"
            run_dir = exp_dir / run_dir
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            
            if not isinstance(run.num_seeds, int):
                raise ValueError("Number of seeds must be of type int")
            
            seeds = GLUE.generate_seeds(run.num_seeds)
            
            for s, seed in enumerate(seeds):
                start_time = time.time()
                run_state = State(run_dir = run_dir)
                logger.info(f"Running with Seed = {seed}")
                try:
                    for p, problem in enumerate(run.problems):
                        logger.info(
                            f"Running Problem number {p+1}  out of {len(run.problems)}"
                            f" with seed number {s+1} out of {len(seeds)}"
                        )
                        score = GLUE.run(
                            problem = problem,
                            run_dir = run_dir,
                            model_dir = model_dir,
                            budget_type = run.budget_type,
                            budget = run.budget,
                            seed = seed,
                            best_run_score = best_score
                            )
                        if score > best_score:
                            best_score = score
                            print(f"New best score: {best_score}")
                            if not problem.problem_statement.hyperparameters:
                                best_model_name = (
                                    f"{problem.problem_statement.optimizer.name}_"
                                    f"{problem.problem_statement.objective_function._model.name}"
                                )
                            else:
                                best_model_name = (
                                    f"{problem.problem_statement.optimizer.name}_"
                                    f"{list(problem.problem_statement.hyperparameters.keys())[0]}_"
                                    f"{list(problem.problem_statement.hyperparameters.values())[0]}_"
                                    f"{problem.problem_statement.objective_function._model.name}"
                                )
                        logger.info(f"Current best score: {best_score} is from Model: {best_model_name}")
                except Exception as e:
                    logger.error(f"Error in Run: {i}, Seed: {seed}")
                    run_state.set_state(run_state.CRASHED)
                    raise e
                except KeyboardInterrupt:
                    logger.error(f"Interrupted Run: {i}, Seed: {seed}")
                    run_state.set_state(run_state.CRASHED)
                    raise SystemExit
                end_time = time.time()
                run_state.set_state(run_state.COMPLETE)
                logger.info(f"Time taken for Run: {i}, Seed: {seed}: {end_time - start_time} seconds")
        return exp_dir, best_score, best_model_name
