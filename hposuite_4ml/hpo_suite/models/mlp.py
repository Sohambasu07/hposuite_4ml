from __future__ import annotations

from sklearn.neural_network import MLPRegressor
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


class MLP(BaseRegressor):
    name = "MLP_Regressor"
    fidelity = "max_iter"
    
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
                    "hidden_layer_sizes",
                    lower=16,
                    upper=1024,
                    default_value=64,
                    log=True
                ),
                CategoricalHyperparameter(
                    "activation",
                    choices=["tanh", "relu"],
                    default_value="relu"
                ),
                UniformIntegerHyperparameter(
                    "batch_size",
                    lower=4,
                    upper=256,
                    default_value=32,
                    log=True
                ),
                # Constant(
                #     "batch_size",
                #     value="auto"
                # ),
                Constant(
                    "solver",
                    value="adam"
                ),
                UniformFloatHyperparameter(
                    "alpha",
                    lower=1e-8,
                    upper=1.0,
                    default_value=1e-3,
                    log=True
                ),
                CategoricalHyperparameter(
                    "learning_rate",
                    choices=["constant", "invscaling", "adaptive"],
                    default_value="constant"
                ),
                UniformFloatHyperparameter(
                    "learning_rate_init",
                    lower=1e-5,
                    upper=1.0,
                    default_value=1e-3,
                    log=True
                ),
                Constant(
                    "tol",
                    value=1e-4
                ),
                # UniformFloatHyperparameter(
                #     # NOTE: Only used when solver='sgd'
                # 
                #     "momentum",
                #     lower=0.1,
                #     upper=0.9,
                #     default_value=0.9
                # ),
                # CategoricalHyperparameter(
                #     # NOTE: Only used when solver='sgd'
                # 
                #     "nesterovs_momentum",
                #     choices=[True, False],
                #     default_value=True
                # ),
                Constant(
                    "beta_1",
                    value=0.9
                ),
                Constant(
                    "beta_2",
                    value=0.999
                ),
                Constant(
                    "epsilon",
                    value=1e-8
                )
            ]
        )
        return cs

    @classmethod
    def create_fidelity_space(cls) -> list[int]:
        fs = list(range(1, 243))
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
        self._model: BaseEstimator = MLPRegressor(**config)
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
    