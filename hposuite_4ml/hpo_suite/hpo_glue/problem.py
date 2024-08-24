from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hposuite_4ml.hpo_suite.hpo_glue.optimizer import Optimizer
    from hposuite_4ml.hpo_suite.hpo_glue.obj_func import ObjectiveFunction

class ProblemStatement:
    
    name: str
    """The name of the problem statement. This is used to identify the problem statement
    in the results and in the filesystem"""

    optimizer: Optimizer
    """The optimizer to use for this problem statement"""

    objective_function: ObjectiveFunction
    """The objective function to optimize"""

    hyperparameters: dict[str, Any]
    """The hyperparameters to use for the optimizer"""


    def __init__(
        self,
        objective_function: ObjectiveFunction,
        optimizer: Optimizer,
        hyperparameters: dict[str, Any] = {},
    ) -> None:
                
        self.optimizer = optimizer
        self.objective_function = objective_function
        self.hyperparameters = hyperparameters
        if self.hyperparameters is None:
            self.hyperparameters = {}
        self.name = f"{objective_function.name}_{optimizer.name}_"
        if bool(self.hyperparameters):
            self.name += f"{list(self.hyperparameters.keys())[0]}-{list(self.hyperparameters.values())[0]}"


class Problem:

    problem_statement: ProblemStatement
    """The Problem Statements to optimize over"""

    objectives: str | list[str]
    """The metric over which to optimize the ObjectiveFunction.

    * str -> single objective
    * list[str] -> multi-objective
    """

    minimize: bool | list[bool]
    """Whether to minimize or maximize the objective value. One per objective"""

    fidelities: str | list[str] | None
    """The fidelity parameter of the model or an explicitly entered dataset fidelity
    """


    def __init__(
        self,
        problem_statement: ProblemStatement,
        objectives: str | list[str],
        minimize: bool | list[bool] = True,
        fidelities: str | list[str] | None = None,
    ) -> None:
                
        self.problem_statement = problem_statement
        self.objectives = objectives    # TODO: Multiobjective not yet supported
        self.minimize = minimize        
        self.fidelities = fidelities    # TODO: PLACEHOLDER. Manyfidelity not yet supported
       
    @property
    def is_multiobjective(self) -> bool:
        return isinstance(self.objectives, list)
        