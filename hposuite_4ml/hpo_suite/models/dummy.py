from __future__ import annotations

from sklearn.dummy import DummyRegressor
from ConfigSpace import (
    ConfigurationSpace, 
    UniformFloatHyperparameter, 
    CategoricalHyperparameter
)
from typing import Any, TYPE_CHECKING
import pickle

from hposuite_4ml.hpo_suite.hpo_glue.base_regressor import BaseRegressor

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    import numpy as np


class Dummy(BaseRegressor):
    name = "Dummy_Regressor"
    fidelity = None
    
    def __init__(self) -> None:
        cls = self.__class__
        cls.fidelity_space = None
        cls.default_fidelity = None
        cls.config_space = cls._create_config_space()

    @classmethod
    def _create_config_space(cls) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        cs.add(
            [
                CategoricalHyperparameter(
                    "strategy",
                    choices=["mean", "median", "quantile", "constant"],
                    default_value="mean"
                ),
                UniformFloatHyperparameter(
                    "constant",
                    lower=-2E31,
                    upper=2E31-1,
                    default_value=None
                ),
                UniformFloatHyperparameter(
                    "quantile",
                    lower=0.1,
                    upper=1.0,
                    default_value=None
                )
            ]
        )
        return cs

    def fit(
        self,
        seed: int,
        config: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        warm_start: bool = False,
        ws_model_name: str = None
    ) -> BaseRegressor:
        self._model: BaseEstimator = DummyRegressor(**config)
        if warm_start:
            # logger.info("Warm start is enabled.")
            if ws_model_name is None:
                raise ValueError("Warm start is enabled but no model name is provided.")
            with open(ws_model_name, "rb") as f:
                self._model = pickle.load(f)
                self._model.set_params(**config)
                
        self._model.fit(X_train, y_train)
        return self

    def predict(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        return self._model.predict(X_test)
    