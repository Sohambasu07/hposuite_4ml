from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hposuite_4ml.hpo_suite.hpo_glue.config import Config

class Query:
    config: Config  
    """ The config to evaluate """
    
    fidelity: Any | dict[str, Any]  
    """ What fidelity to evaluate at """

    def __init__(self, config: Config, fidelity: Any | dict[str, Any]):
        self.config = config
        self.fidelity = fidelity