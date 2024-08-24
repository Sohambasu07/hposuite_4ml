"""
The ObjectiveFunction class combines a specific model, metric(s) and train-test data
to first fit the model on the train dataset and then evaluate it on the test dataset
using the metric(s) defined.
"""


from __future__ import annotations

from sklearn.metrics import r2_score
import numpy as np
from typing import TYPE_CHECKING
import os

from hposuite_4ml.hpo_suite.hpo_glue.result import Result

if TYPE_CHECKING:
    from hposuite_4ml.hpo_suite.hpo_glue.query import Query
    from hposuite_4ml.hpo_suite.hpo_glue.base_regressor import BaseRegressor

METRICS = {
    "r2": r2_score,
}

class ObjectiveFunction:
    def __init__(
        self,
        name: str,
        dataset_name: str,
        train_dataset: tuple[np.ndarray, np.ndarray],
        test_dataset: tuple[np.ndarray, np.ndarray],
        model: BaseRegressor,
        metric: str = "r2",
        warm_start: bool = False,
        ws_model_name: str = None
    ) -> None:
        self.name = name
        self.dataset_name = dataset_name
        self.X_train, self.y_train = train_dataset
        self.X_test, self.y_test = test_dataset

        # Convert p.Dataframe and pd.Series to numpy arrays
        self.X_train = self.X_train.to_numpy()
        self.y_train = self.y_train.to_numpy()
        self.X_test = self.X_test.to_numpy()
        self.y_test = self.y_test.to_numpy()

        self._model = model
        self._query = None
        self._metric = metric
        self._warm_start = warm_start
        self._ws_model_name = ws_model_name

    def fit(
        self,
        seed: int
    ) -> None:
        seed = int(seed)
        ws = self._warm_start
        if self._warm_start and os.path.exists(self._ws_model_name) is False:
            ws = False
        config = self._query.config.values
        if self._query.fidelity is not None:
            fid = {self._model.fidelity: self._query.fidelity}
            config.update(fid)
        self._model = self._model.fit(
            seed=seed,
            config=config,
            X_train=self.X_train,
            y_train=self.y_train,
            warm_start=ws,
            ws_model_name=self._ws_model_name
        )

    def evaluate(
        self
    ) -> Result:

        if self._query is None:
            raise ValueError("Cannot evaluate without a query!")
        preds = self._model.predict(self.X_test)
        score = METRICS[self._metric](self.y_test, preds)
        result = {
            self._metric: score
        }

        return Result(
            query=self._query,
            result=result
        )

    def __call__(
        self,
        seed: int,
        query: Query
    ) -> Result:
        self._query = query
        self.fit(seed)
        return self.evaluate()