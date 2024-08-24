from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace
    import numpy as np


class BaseRegressor(ABC):
    name: str
    fidelity: str | None
    fidelity_space: list[str] | list[float] | None
    default_fidelity: int | float | None
    config_space: ConfigurationSpace

    def __init__(self) -> None:
        ...

    def fit(
        self,
        seed: int,
        config: dict[str, any],
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> BaseRegressor:
        ...

    def predict(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        ...