from __future__ import annotations
from typing import Any

from ConfigSpace import Configuration

class Config:
    id: str  
    """ Some unique identifier """

    values: dict[str, Any]  
    """ The actual config values to evaluate """

    def __init__(self, id: str, values: Configuration | dict[str, Any]):
        self.id = id
        if isinstance(values, Configuration):
            self.values = values.get_dictionary()
        else:
            self.values = values