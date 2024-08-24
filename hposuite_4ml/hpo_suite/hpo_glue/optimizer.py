from __future__ import annotations
from abc import ABC
from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from hposuite_4ml.hpo_suite.hpo_glue.problem import Problem
    from hposuite_4ml.hpo_suite.hpo_glue.query import Query
    from hposuite_4ml.hpo_suite.hpo_glue.result import Result
    from pathlib import Path

class Optimizer(ABC):
    """ Defines the common interface for Optimizers """

    name: ClassVar[str]
    supports_manyfidelity: ClassVar[bool] = False
    supports_multifidelity: ClassVar[bool] = False
    supports_multiobjective: ClassVar[bool] = False
    supports_tabular: ClassVar[bool] = False

    def __init__(
        self,
        problem: Problem,
        working_directory: Path,
        seed: int | None
    ) -> None:
        ...

    def ask(self) -> Query:
        """Ask the optimizer for a new config to evaluate"""
        ...

    def tell(self, result: Result) -> None:
        """Tell the optimizer the result of the query"""
        ...