from __future__ import annotations

from lightgbm import LGBMRegressor
from ConfigSpace import (
    ConfigurationSpace, 
    UniformFloatHyperparameter, 
    UniformIntegerHyperparameter
)
from typing import Any, TYPE_CHECKING
import pickle

from hposuite_4ml.hpo_suite.hpo_glue.base_regressor import BaseRegressor

import warnings
warnings.filterwarnings("ignore")


if TYPE_CHECKING:
    from sklearn.base import RegressorMixin
    import numpy as np


class LGBM(BaseRegressor):
    name = "LightGBM_Regressor"
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
                UniformIntegerHyperparameter(
                    "num_leaves",
                    lower=4,
                    upper=2**10,
                    default_value=4,
                    log=True
                ),
                UniformIntegerHyperparameter(
                    "max_depth",
                    lower=-1,
                    upper=10,
                    default_value=-1
                ),
                UniformFloatHyperparameter(
                    "learning_rate",
                    lower=2**-10,
                    upper=1.0,
                    default_value=0.1,
                    log=True
                ),
                UniformIntegerHyperparameter(
                    "min_data_in_leaf",
                    lower=2,
                    upper=2**7+1,
                    default_value=20,
                    log=True
                ),
              #  UniformIntegerHyperparameter(
              #      "max_bin",
              #      lower=3,
              #      upper=11,
              #      default_value=8
              #  ),
                UniformFloatHyperparameter(
                    "feature_fraction",
                    lower=0.01,
                    upper=1.0,
                    default_value=1.0
                ),
                UniformIntegerHyperparameter(
                    "min_child_samples",
                    lower=1,
                    upper=50,
                    default_value=20
                ),
                UniformFloatHyperparameter(
                    "reg_alpha",    #NOTE: alias: lambda_l1, l1_regularization
                    lower=2**-10,
                    upper=2**10,
                    default_value=2**-10,
                    log=True
                ),
                UniformFloatHyperparameter(
                    "reg_lambda",   #NOTE: alias: lambda_l2, l2_regularization
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
        other_params = {
            "random_state": seed,
            "device_type": "cpu",
            "verbosity": -1,
        }
        config.update(other_params)
        self._model: RegressorMixin = LGBMRegressor(**config)
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
    
