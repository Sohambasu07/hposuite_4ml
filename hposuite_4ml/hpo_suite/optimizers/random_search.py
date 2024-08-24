from __future__ import annotations

from ConfigSpace import ConfigurationSpace, Configuration
from pathlib import Path
from typing import TYPE_CHECKING
import random

from hposuite_4ml.hpo_suite.hpo_glue.optimizer import Optimizer
from hposuite_4ml.hpo_suite.hpo_glue.config import Config
from hposuite_4ml.hpo_suite.hpo_glue.query import Query

if TYPE_CHECKING:
    from hposuite_4ml.hpo_suite.hpo_glue.problem import Problem
    from hposuite_4ml.hpo_suite.hpo_glue.result import Result
    

class RandomSearch(Optimizer):
    name = "RandomSearch"
    # supports_multifidelity = True

    def __init__(self, 
                 problem: Problem,
                 working_directory: Path,
                 seed: int | None = None):
        """ Create a Random Search Optimizer instance for a given problem """

        if isinstance(problem.objectives, list):
            raise NotImplementedError("# TODO: Implement multiobjective for RandomSearch")
        
        if isinstance(problem.fidelities, list):
            raise NotImplementedError("# TODO: Manyfidelity not yet implemented for RandomSearch!")

        self.problem = problem
        self.config_space: ConfigurationSpace = self.problem.problem_statement.objective_function._model.config_space
        self.model_fidelity: str | None = self.problem.problem_statement.objective_function._model.fidelity
        self.fidelity_space: list[int] | list[float] = self.problem.problem_statement.objective_function._model.fidelity_space
        self.objectives = self.problem.objectives
        if seed is None:
            seed = self.rng.randint(0, 2**31-1)  # Big number so we can sample 2**31-1 possible configs
        self.seed = int(seed)
        self.config_space.seed(self.seed)
        self.rng = random.Random(self.seed)
        print(f"Random Search seed: {self.seed}")
        self.minimize = self.problem.minimize
        self.is_multiobjective = self.problem.is_multiobjective
        
    def get_config(self, num_configs: int) -> Configuration | list[Configuration]:
        """ Sample a random config or a list of configs from the configuration space """
        config = self.config_space.sample_configuration(num_configs)
        return config
        
    def ask(self,   
            config_id: str | None = None) -> Query:
        """ Ask the optimizer for a new config to evaluate """

        
        fidelity = self.problem.problem_statement.objective_function._model.default_fidelity
        
        # # Randomly sampling from fidelity space for multifidelity
        # if self.model_fidelity is not None:
        #     fidelity = self.rng.choice(self.fidelity_space)

        config = self.get_config(1)
        return Query(Config(config_id, config), fidelity)
    
    def tell(self, result: Result) -> None:
        """ Tell the optimizer the result of the query """

        cost = result.result[self.problem.objectives]
        if self.minimize is False:
            cost = -cost