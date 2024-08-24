from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from ConfigSpace import (
    ConfigurationSpace, 
    UniformFloatHyperparameter, 
    UniformIntegerHyperparameter, 
    CategoricalHyperparameter,
    Constant
)
from typing import Any, TYPE_CHECKING
import pickle

from hposuite_4ml.hpo_suite.hpo_glue.base_regressor import BaseRegressor

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    import numpy as np


class RF(BaseRegressor):
    name = "RF_Regressor"
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
                # CategoricalHyperparameter(
                #     # NOTE: Takes too long to run
                
                #     "criterion",
                #     choices=["squared_error", "absolute_error", "friedman_mse"]
                # ),
                UniformFloatHyperparameter(
                    "max_features",
                    lower=0.1,
                    upper=1.0,
                    default_value=1.0
                ),
                # UniformIntegerHyperparameter(
                #     # NOTE: Removing this increases runtime 
                
                #     "max_depth",
                #     lower=1,
                #     upper=10,
                #     default_value=3
                #     log=True
                # ),
                UniformIntegerHyperparameter(
                    "min_samples_split",
                    lower=2,
                    upper=20,
                    default_value=2
                ),
                UniformIntegerHyperparameter(
                    "min_samples_leaf",
                    lower=1,
                    upper=20,
                    default_value=1
                ),
                Constant(
                    "min_weight_fraction_leaf",
                    value=0.0
                ),
                UniformIntegerHyperparameter(
                    # NOTE: Removing this increases runtime
                    "max_leaf_nodes",
                    lower=2,
                    upper=100,
                    default_value=20
                ),
                Constant(
                    "min_impurity_decrease",
                    value=0.0
                ),
                CategoricalHyperparameter(
                    "bootstrap",
                    choices=[True, False]
                )
            ]
        )
        return cs

    @classmethod
    def create_fidelity_space(cls) -> list[int]:
        fs = list(range(1, 512))
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
        config.update(rs)
        self._model: BaseEstimator = RandomForestRegressor(**config)
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
    