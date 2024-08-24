from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hposuite_4ml.hpo_suite.hpo_glue.query import Query

class Result:
    """The result of a query from an objective function."""

    query: Query
    """The query that generated this result"""


    result: dict[str, Any]
    """Everything returned by the objective function for a given query."""

    def __init__(self, query: Query, result: dict[str, Any]):
        self.query = query
        self.result = result