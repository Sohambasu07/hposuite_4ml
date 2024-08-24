from __future__ import annotations

import os
from ConfigSpace import ConfigurationSpace
from pathlib import Path
from typing import TYPE_CHECKING
from dehb import DEHB
import datetime

from hposuite_4ml.hpo_suite.hpo_glue.optimizer import Optimizer
from hposuite_4ml.hpo_suite.hpo_glue.config import Config
from hposuite_4ml.hpo_suite.hpo_glue.query import Query

if TYPE_CHECKING:
    from hposuite_4ml.hpo_suite.hpo_glue.problem import Problem
    from hposuite_4ml.hpo_suite.hpo_glue.result import Result


class DEHB_Optimizer(Optimizer):
    name = "DEHB"
    supports_multifidelity = True

    def __init__(
        self,
        problem: Problem,
        working_directory: Path,
        seed: int | None = None,
        eta: int = 3
    ):
        """ Create a DEHB Optimizer instance for a given Problem """

        
        self.problem = problem
        self.config_space: ConfigurationSpace = self.problem.problem_statement.objective_function._model.config_space
        self.fidelity_space: list[int] | list[float] | None = self.problem.problem_statement.objective_function._model.fidelity_space
        self.objectives: str | list[str] = self.problem.objectives
        self.minimize = self.problem.minimize
        self.seed = int(seed)
        # TODO: Set seed if seed is None
        self.config_space.seed(self.seed)

        if os.path.exists(working_directory) is False:
            os.makedirs(working_directory)

        min_budget = None
        max_budget = None

        if self.fidelity_space is not None:
            min_budget = self.fidelity_space[0]
            max_budget = self.fidelity_space[-1]

        self.is_multiobjective = self.problem.is_multiobjective

        self.dehb = DEHB(
            cs = self.config_space,
            min_fidelity = min_budget,
            max_fidelity = max_budget,
            seed = self.seed,
            eta = eta,
            n_workers = 1,
            output_path = working_directory
        )

        self.info = None


    def ask(
        self,
        config_id: str | None =  None
    ) -> Query:
        """ Ask DEHB for a new config to evaluate """

        self.info = self.dehb.ask()
        config = Config(
            id = f"{self.seed}_{datetime.time()}",
            values = self.info["config"]
        )
        fidelity = self.info["fidelity"]
        
        if isinstance(fidelity, float):
            fidelity = round(fidelity)

        return Query(config=config, fidelity=fidelity)
    
    def tell(
        self,
        result: Result
    ) -> None:
        """ Tell DEHB the result of the query """

        cost = result.result[self.objectives]   #Not considering Multiobjective for now
        if self.minimize is False:
            cost = -cost
        
        dehb_result = {
            "fitness": float(cost), # Actual objective value
            "cost": self.info['fidelity'] # TODO: time or fidelity cost
        }
        self.dehb.tell(self.info, dehb_result)