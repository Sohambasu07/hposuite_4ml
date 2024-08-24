from __future__ import annotations
from typing import Any, TYPE_CHECKING
from pathlib import Path
import os
import pandas as pd

if TYPE_CHECKING:
    from hposuite_4ml.hpo_suite.hpo_glue.result import Result
    

class History:
    """Class for storing the history of an optimizer run."""

    results: list[Result]

    def __init__(self) -> None:
        self.results = []

    def add(self, result: Result) -> None:
        self.results.append(result)

    def df(self, columns) -> pd.DataFrame:
        """Return the history as a pandas DataFrame"""

        report = []

        for res in self.results:
            config = res.query.config.values
            id = res.query.config.id
            fidelity = res.query.fidelity
            result = res.result
            report.append([id, fidelity])
            report[-1].extend([val for key, val in config.items()])
            report[-1].extend([val for key, val in result.items()])

        hist_df = pd.DataFrame(report, columns=columns)
        return hist_df

    def _save(
            self, 
            report: pd.DataFrame, 
            runsave_dir: Path,
            dataset_name: str,
            model_name: str,
            optimizer_name: str,
            optimizer_hyperparameters: dict[str, Any],
            seed: int
    ) -> None:
        """ Save the history of the run and along with some metadata """
        
        optimizer_hyperparameters = optimizer_hyperparameters if bool(optimizer_hyperparameters) else ''
        filename = f"{dataset_name}_{optimizer_name}_{optimizer_hyperparameters}_{model_name}"
        filesave_dir = runsave_dir / dataset_name/ optimizer_name / str(seed)
        if os.path.exists(filesave_dir) is False:
            os.makedirs(filesave_dir)
        report.convert_dtypes().to_parquet(filesave_dir / f"report_{filename}.parquet", index=False)
