from __future__ import annotations

import os
from ConfigSpace import (
    ConfigurationSpace,
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant
)
from ConfigSpace.hyperparameters import OrdinalHyperparameter
from pathlib import Path
import datetime
from typing import TYPE_CHECKING

from syne_tune.optimizer.baselines import (
    BayesianOptimization,
    SyncBOHB,
    KDE,
    ASHA, 
    BOHB,
    DEHB
)
from syne_tune.backend.trial_status import Trial, Status, TrialResult                   # noqa
from syne_tune.config_space import uniform, loguniform, ordinal, choice, randint

from hposuite_4ml.hpo_suite.hpo_glue.optimizer import Optimizer
from hposuite_4ml.hpo_suite.hpo_glue.config import Config
from hposuite_4ml.hpo_suite.hpo_glue.query import Query
if TYPE_CHECKING:
    from hposuite_4ml.hpo_suite.hpo_glue.problem import Problem
    from hposuite_4ml.hpo_suite.hpo_glue.result import Result
    from syne_tune.optimizer.scheduler import TrialScheduler


def configspace_to_synetune_configspace(config_space: ConfigurationSpace) -> dict:
    """Convert ConfigSpace to SyneTune config_space"""
    if not isinstance(config_space, ConfigurationSpace):
        raise ValueError("config_space must be of type ConfigSpace.ConfigurationSpace")
    synetune_cs = {}
    for hp in config_space.get_hyperparameters():
        if isinstance(hp, OrdinalHyperparameter):
            synetune_cs[hp.name] = ordinal(hp.sequence)
        elif isinstance(hp, CategoricalHyperparameter):
            synetune_cs[hp.name] = choice(hp.choices) # choice.weights in  ConfigSpace -> check SyneTune
        elif isinstance(hp, UniformIntegerHyperparameter):
            synetune_cs[hp.name] = randint(hp.lower, hp.upper) # check for logscale (hp.log)
        elif isinstance(hp, UniformFloatHyperparameter):
            if hp.log:
                synetune_cs[hp.name] = loguniform(hp.lower, hp.upper)
            else:
                synetune_cs[hp.name] = uniform(hp.lower, hp.upper)
        elif isinstance(hp, Constant):
            synetune_cs[hp.name] = hp.value
        else:
            raise ValueError(f"Hyperparameter {hp.name} of type {type(hp)} is not supported")
        
    return synetune_cs


class SyneTuneOptimizer(Optimizer):
    name = "SyneTune_BO"

    def __init__(
        self,
        problem: Problem,
        working_directory: Path,
        seed: int | None = None
    ):
        """ Create a SyneTune Optimizer instance for a given Problem

        (Curretly running Bayesian Optimization as the base/default optimizer)
        """

        self.problem = problem
        self.config_space: ConfigurationSpace = self.problem.problem_statement.objective_function._model.config_space
        self.fidelity_space: list[int] | list[float] | None = self.problem.problem_statement.objective_function._model.fidelity_space
        self.objectives: str | list[str] = self.problem.objectives
        self.seed = seed 

        self.synetune_cs = configspace_to_synetune_configspace(self.config_space)
        if self.__class__.supports_multifidelity:
            self.synetune_cs[
                self.problem.problem_statement.objective_function._model.fidelity
            ] = int(self.fidelity_space[-1])
        

        if os.path.exists(working_directory) is False:
            os.makedirs(working_directory)

        self.bscheduler: TrialScheduler = BayesianOptimization(
            config_space=self.synetune_cs,
            metric=self.objectives,
            mode = "min" if self.problem.minimize else "max",
            random_seed = self.seed,
        )
        
        self.trial_counter = 0
        self.trial = None
        self.hisory: dict[str, tuple[Config, Result]] = {}

    def ask(
            self,
            config_id: str | None = None
    ) -> Query:
        
        """Get a configuration from the optimizer"""
        
        trial_suggestion = self.bscheduler.suggest(self.trial_counter)
        self.trial = Trial(
            trial_id=str(self.trial_counter),
            config = trial_suggestion.config,
            creation_time = datetime.datetime.now()
        )
        config = Config(
            id = str(self.trial_counter),
            values = self.trial.config
        )

        fidelity = None
        if self.__class__.supports_multifidelity:
            fidelity = trial_suggestion.config.pop(self.problem.problem_statement.objective_function._model.fidelity)
        self.hisory[config.id] = (config, fidelity)

        if fidelity is None:
            fidelity = self.problem.problem_statement.objective_function._model.default_fidelity

        return Query(config = config, fidelity = fidelity)
    
    def tell(
            self,
            result: Result
    ) -> None:
        """Update the SyneTune Optimizer with the result of a Query"""

        results_obj_dict = {key: result.result[key] for key in result.result.keys() if key == self.objectives}
        if self.__class__.supports_multifidelity:
            results_obj_dict[self.problem.problem_statement.objective_function._model.fidelity] = result.query.fidelity

        cost = result.result[self.objectives]
        if self.problem.minimize:
            cost = -cost

        self.trial_counter += 1

        self.bscheduler.on_trial_complete(
            trial = self.trial, 
            result=results_obj_dict
        )


class SyneTune_KDE(SyneTuneOptimizer):
    name = "SyneTune_KDE"

    def __init__(
            self,
            problem: Problem,
            working_directory: Path,
            seed: int | None = None,
            eta: int = 3,
    ):
        """ Create a SyneTune KDE instance for a given Problem """

        super().__init__(problem, working_directory, seed)

        self.bscheduler = KDE(
            config_space=self.synetune_cs,
            mode = "min" if self.problem.minimize else "max",
            metric = self.objectives,
            random_seed = seed
        )


class SyneTune_ASHA(SyneTuneOptimizer):
    name = "SyneTune_ASHA"
    supports_multifidelity = True

    def __init__(
            self,
            problem: Problem,
            working_directory: Path,
            seed: int | None = None,
            eta: int = 3,
    ):
        """ Create a SyneTune ASHA instance for a given Problem """

        super().__init__(problem, working_directory, seed)

        self.bscheduler = ASHA(
            config_space=self.synetune_cs,
            type="stopping",
            max_t = self.fidelity_space[-1],
            resource_attr = problem.problem_statement.objective_function._model.fidelity,
            mode = "min" if self.problem.minimize else "max",
            metric = self.objectives,
            grace_period=1,
            reduction_factor = eta,
            random_seed = seed
        )


class SyneTune_BOHB(SyneTuneOptimizer):
    name = "SyneTune_BOHB"
    supports_multifidelity = True

    def __init__(
            self,
            problem: Problem,
            working_directory: Path,
            seed: int | None = None,
            eta: int = 3,
    ):
        """ Create a SyneTune BOHB instance for a given Problem """

        super().__init__(problem, working_directory, seed)

        self.bscheduler = BOHB(
            config_space=self.synetune_cs,
            type="stopping",
            max_t = self.fidelity_space[-1],
            resource_attr = problem.problem_statement.objective_function._model.fidelity,
            mode = "min" if self.problem.minimize else "max",
            metric = self.objectives,
            grace_period=1,
            reduction_factor = eta,
            random_seed = seed
        )


class SyneTune_SyncBOHB(SyneTuneOptimizer):
    name = "SyneTune_SyncBOHB"
    supports_multifidelity = True

    def __init__(
            self,
            problem: Problem,
            working_directory: Path,
            seed: int | None = None,
            eta: int = 3,
    ):
        """ Create a SyneTune SyncBOHB instance for a given Problem """

        super().__init__(problem, working_directory, seed)

        self.bscheduler = SyncBOHB(
            config_space=self.synetune_cs,
            resource_attr = problem.problem_statement.objective_function._model.fidelity,
            max_resource_level = self.fidelity_space[-1],
            mode = "min" if self.problem.minimize else "max",
            metric = self.objectives,
            grace_period=1,
            reduction_factor = eta,
            random_seed = seed
        )

class SyneTune_DEHB(SyneTuneOptimizer):
    name = "SyneTune_DEHB"
    supports_multifidelity = True

    def __init__(
            self,
            problem: Problem,
            working_directory: Path,
            seed: int | None = None,
            eta: int = 3,
    ):
        """ Create a SyneTune DEHB instance for a given Problem """

        super().__init__(problem, working_directory, seed)

        self.bscheduler = DEHB(
            config_space=self.synetune_cs,
            resource_attr = problem.problem_statement.objective_function._model.fidelity,
            max_resource_level = self.fidelity_space[-1],
            mode = "min" if self.problem.minimize else "max",
            metric = self.objectives,
            grace_period=1,
            reduction_factor = eta,
            random_seed = seed
        )