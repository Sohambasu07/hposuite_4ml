from __future__ import annotations

from ConfigSpace import (
    ConfigurationSpace,
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant
)
from ConfigSpace.hyperparameters import OrdinalHyperparameter
from pathlib import Path
import logging
from typing import TYPE_CHECKING

import optuna
from optuna.samplers import *               # noqa  
from optuna.distributions import (
    CategoricalDistribution as Cat,
    FloatDistribution as Float,
    IntDistribution as Int
)

from hposuite_4ml.hpo_suite.hpo_glue.optimizer import Optimizer
from hposuite_4ml.hpo_suite.hpo_glue.config import Config
from hposuite_4ml.hpo_suite.hpo_glue.query import Query

if TYPE_CHECKING:
    from hposuite_4ml.hpo_suite.hpo_glue.problem import Problem
    from hposuite_4ml.hpo_suite.hpo_glue.result import Result


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

samplers = {
    "TPE": TPESampler,                  # noqa
    "Random": RandomSampler,            # noqa
    "Grid": GridSampler,                # noqa
    "CMAES": CmaEsSampler,              # noqa
    "GP": GPSampler,                    # noqa
}


class OptunaOptimizer(Optimizer):
    name = "Optuna"

    def __init__(
        self,
        problem: Problem,
        working_directory: Path,
        sampler: str | None = None,
        seed: int | None = None # Only way to add seed to Optuna is by defining a sampler
    ):
        """ Create a SMAC Optimizer instance for a given Problem """

        if isinstance(problem.objectives, list):
            raise NotImplementedError("Multiobjective not implemented for Optuna!")
        
        if isinstance(problem.fidelities, list):
            raise NotImplementedError("Manyfidelity not yet implemented for Optuna!")
        
        self.problem = problem
        self.config_space: ConfigurationSpace = self.problem.problem_statement.objective_function._model.config_space
        self.model_fidelity: str | None = self.problem.problem_statement.objective_function._model.fidelity
        self.fidelity_space: list[int] | list[float] | None = self.problem.problem_statement.objective_function._model.fidelity_space
        self.objectives: str | list[str] = self.problem.objectives
        # if seed is None:
        #     seed = -1
        self.seed = seed

        self.study = optuna.create_study(
            direction = "minimize" if self.problem.minimize else "maximize"
        )

        if sampler is not None and sampler not in samplers:
            raise ValueError(f"Sampler {sampler} not supported by Optuna!")

        if sampler is not None:
            self.study.sampler = samplers.get(sampler)(seed=self.seed)

        self.distributions = configspace_to_optuna_distributions( 
            self.config_space
        )
        self.counter = 0

    def ask(
        self, 
        config_id: str | None =  None
    ) -> Query:
        self.trial = self.study.ask(self.distributions)
        config = Config(
            id=f"{self.seed}_{self.counter}",
            values=self.trial.params,
        )
        self.counter += 1
        fidelity = self.problem.problem_statement.objective_function._model.default_fidelity
        return Query(config=config, fidelity = fidelity)

    def tell(
        self, 
        result: Result
    ) -> None:
        self.study.tell(self.trial, result.result[self.problem.objectives])


def configspace_to_optuna_distributions(config_space: ConfigurationSpace) -> dict:
    if not isinstance(config_space, ConfigurationSpace):
        raise ValueError("Need search space of type ConfigSpace.ConfigurationSpace}")
    optuna_space = dict()
    for hp in config_space.get_hyperparameters():
        if isinstance(hp, UniformIntegerHyperparameter):
            optuna_space[hp.name] = Int(hp.lower, hp.upper, log=hp.log)
        elif isinstance(hp, UniformFloatHyperparameter):
            optuna_space[hp.name] = Float(hp.lower, hp.upper, log=hp.log)
        elif isinstance(hp, CategoricalHyperparameter):
            optuna_space[hp.name] = Cat(hp.choices)
        elif isinstance(hp, Constant):
            if isinstance(hp.value, (int, float)):
                optuna_space[hp.name] = Float(hp.value, hp.value)
            else:
                print(f"hp.name: {hp.name}")
                print(f"hp.value: {hp.value}")
                optuna_space[hp.name] = Cat([hp.value])
        elif isinstance(hp, OrdinalHyperparameter):
            # TODO: handle categoricals better
            optuna_space[hp.name] = Cat(hp.sequence)
        else:
            raise ValueError("Unrecognized type of hyperparameter in ConfigSpace!")
    return optuna_space

