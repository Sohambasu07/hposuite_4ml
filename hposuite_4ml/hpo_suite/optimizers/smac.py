from __future__ import annotations

import os
from ConfigSpace import ConfigurationSpace
from pathlib import Path
from typing import TYPE_CHECKING
from smac import (
    HyperparameterOptimizationFacade as HPOFacade,
    HyperbandFacade as HBFacade,
    BlackBoxFacade as BOFacade,
    Scenario
)
from smac.runhistory.dataclasses import TrialInfo, TrialValue

from hposuite_4ml.hpo_suite.hpo_glue.optimizer import Optimizer
from hposuite_4ml.hpo_suite.hpo_glue.config import Config
from hposuite_4ml.hpo_suite.hpo_glue.query import Query

if TYPE_CHECKING:
    from hposuite_4ml.hpo_suite.hpo_glue.problem import Problem
    from hposuite_4ml.hpo_suite.hpo_glue.result import Result


class SMAC_Optimizer(Optimizer):
    name = "SMAC_HPOFacade"

    def __init__(
        self,
        problem: Problem,
        working_directory: Path,
        seed: int | None = None,
        **kwargs
    ):
        """ Create a SMAC Optimizer instance for a given Problem """

        if isinstance(problem.objectives, list):
            raise NotImplementedError("# TODO: Implement multiobjective for SMAC")
        
        if isinstance(problem.fidelities, list):
            raise NotImplementedError("# TODO: Manyfidelity not yet implemented for SMAC!")
        
        self.problem = problem
        self.config_space: ConfigurationSpace = self.problem.problem_statement.objective_function._model.config_space
        self.model_fidelity: str | None = self.problem.problem_statement.objective_function._model.fidelity
        self.fidelity_space: list[int] | list[float] | None = self.problem.problem_statement.objective_function._model.fidelity_space
        self.objectives: str | list[str] = self.problem.objectives
        if seed is None:
            seed = -1
        self.seed = seed
        self.smac_info : TrialInfo | None = None #No parallel support
        self.smac_val : TrialValue | None = None #No parallel support

        if os.path.exists(working_directory) is False:
            os.makedirs(working_directory)

        min_budget = None
        max_budget = None

        if self.fidelity_space is not None:
            min_budget = self.fidelity_space[0]
            max_budget = self.fidelity_space[-1]

        n_trials = 1
        # if self.problem.budget_type == "n_trials":
        #     n_trials = self.problem.budget

        self.scenario = Scenario(
            name=self.problem.problem_statement.name,
            configspace=self.config_space,
            deterministic=False,
            objectives=self.objectives,
            n_trials=n_trials,
            output_directory=working_directory,
            min_budget=min_budget,
            max_budget=max_budget,
            seed=self.seed
        )
        
        self.is_multiobjective = self.problem.is_multiobjective
        self.minimize = self.problem.minimize

        self.facade = HPOFacade

        self.intensifier = self.facade.get_intensifier(
            self.scenario
        )

        self.acquisition_function = self.facade.get_acquisition_function(
            self.scenario,
            **kwargs
        )

        self.smac = self.facade(
            scenario=self.scenario,
            target_function=lambda seed, budget: None,
            intensifier=self.intensifier,
            overwrite=True,
            acquisition_function=self.acquisition_function
        )
    
    def ask(
        self,
        config_id: str | None =  None
    ) -> Query:
        """ Ask SMAC for a new config to evaluate """
        
        self.smac_info = self.smac.ask()
        config = self.smac_info.config
        budget = self.smac_info.budget
        instance = self.smac_info.instance
        seed = self.smac_info.seed
        fidelity = None

        # if self.smac_info.budget is not None:
        #     fidelity = budget

        if self.__class__.supports_multifidelity is True:
            fidelity = budget

        _config_id = self.intensifier.runhistory.config_ids[config]  #For now using SMAC's own config_id

        config = Config(
            id=f"{_config_id=}_{seed=}_{instance=}",
            values=config.get_dictionary(),
            )
        
        if isinstance(fidelity, float):
            fidelity = round(fidelity)

        if fidelity is None:
            fidelity = self.problem.problem_statement.objective_function._model.default_fidelity

        return Query(config=config, fidelity=fidelity)
    
    def tell(
        self,
        result: Result
    ) -> None:
        """ Tell SMAC the result of the query """
        cost = result.result[self.objectives]   #Not considering Multiobjective for now
        if self.minimize is False:
            cost = -cost
        self.smac_val = TrialValue(cost = cost, time = 0.0)
        self.smac.tell(self.smac_info, self.smac_val)

class SMAC_BO(SMAC_Optimizer):
    name = "SMAC_BO"

    def __init__(
        self,
        problem: Problem,
        working_directory: Path,
        seed: int | None = None,
        xi: float = 0.0
    ):
        super().__init__(problem, working_directory, seed)
        self.facade = BOFacade
        self.intensifier = self.facade.get_intensifier(
            self.scenario,
        )
        self.acquisition_function = self.facade.get_acquisition_function(
            self.scenario,
            xi = xi
        )
        self.smac = self.facade(
            scenario = self.scenario,
            target_function = lambda seed, budget: None,
            intensifier = self.intensifier,
            acquisition_function = self.acquisition_function,
            overwrite = True)


class SMAC_Hyperband(SMAC_Optimizer):
    name = "SMAC_Hyperband"
    supports_multifidelity = True

    def __init__(
        self,
        problem: Problem,
        working_directory: Path,
        eta: int = 3,
        seed: int | None = None
    ):
        super().__init__(problem, working_directory, seed)
        self.facade = HBFacade
        self.intensifier = self.facade.get_intensifier(
            self.scenario,
            eta = eta
        )
        self.smac = self.facade(
            scenario = self.scenario,
            target_function = lambda seed, budget: None,
            intensifier = self.intensifier,
            overwrite = True)
        
