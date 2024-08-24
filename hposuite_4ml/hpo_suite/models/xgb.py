from __future__ import annotations

from xgboost import XGBRegressor
from ConfigSpace import (
    ConfigurationSpace, 
    UniformFloatHyperparameter, 
    UniformIntegerHyperparameter, 
)
from typing import Any, TYPE_CHECKING
import pickle

from hposuite_4ml.hpo_suite.hpo_glue.base_regressor import BaseRegressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin
    import numpy as np


class XGB(BaseRegressor):
    name = "XGBoost_Regressor"
    fidelity = "n_estimators"
    
    def __init__(self) -> None:
        cls = self.__class__
        cls.fidelity_space = cls.create_fidelity_space()
        cls.default_fidelity = cls.fidelity_space[-1]
        cls.config_space = cls._create_config_space()

    @classmethod
    def _create_config_space(cls) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        cs.add(
            [
                UniformFloatHyperparameter(
                    "learning_rate",
                    lower=2**-10,
                    upper=1.0,
                    default_value=0.3,
                    log=True
                ),
                UniformIntegerHyperparameter(
                    "max_depth",
                    lower=1,
                    upper=50,
                    default_value=10
                ),
                UniformIntegerHyperparameter(
                    "max_leaves",
                    lower=4,
                    upper=2**10,
                    default_value=4
                ),
                UniformFloatHyperparameter(
                    "colsample_bytree",
                    lower=0.01,
                    upper=1.0,
                    default_value=1.0,
                    log=True
                ),
                UniformFloatHyperparameter(
                    "reg_alpha",
                    lower=2**-10,
                    upper=2**10,
                    default_value=2**-10,
                    log=True
                ),
                UniformFloatHyperparameter(
                    "reg_lambda",
                    lower=2**-10,
                    upper=2**10,
                    default_value=1.0,
                    log=True
                )
            ]
        )
        return cs

    @classmethod
    def create_fidelity_space(cls) -> list[int]:
        fs = list(range(1, 1001))
        return fs

    def fit(
        self,
        seed: int,
        config: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        warm_start: bool = False,
        ws_model_name: str = None
    ) -> BaseRegressor:
        rs = {"random_state": seed}
        device = {"device": "cpu"}
        config.update(rs)
        config.update(device)

        self._model: RegressorMixin = XGBRegressor(**config)
        if warm_start:
            # logger.info("Warm start is enabled.")
            if ws_model_name is None:
                raise ValueError("Warm start is enabled but no model name is provided.")
            with open(ws_model_name, "rb") as f:
                self._model = pickle.load(f)
                self._model.set_params(**config)

        self._model.fit(X_train, y_train, eval_set=None)
        return self

    def predict(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:
        return self._model.predict(X_test)
    