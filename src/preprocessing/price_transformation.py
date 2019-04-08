"""Transformers which operate on a single column."""

from typing import List, Tuple, Type, TypeVar, Union, Callable, Sequence

import numpy as np
import pandas as pd

from .base import BasePandasTransformer


class LogReturns(BasePandasTransformer):
    """Take the log return."""

    def __init__(self, columns: Sequence[str]):
        super().__init__(columns)

    def fit(self, X: pd.DataFrame, y=None) -> Type['LogReturns']:
        super().prepare_to_fit(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        log: pd.DataFrame = np.log(X.loc[:, self.columns])
        result = log.diff(periods=1)
        result.columns = [f"{old_name}_log_ret" for old_name in result.columns]

        return result


class PercentReturns(BasePandasTransformer):
    """Take the percentage returns."""

    def __init__(self, columns: Sequence[str]):
        super().__init__(columns)

    def fit(self, X: pd.DataFrame, y=None) -> Type['PercentReturns']:
        self.prepare_to_fit(X)

        vals = X.loc[:, self.columns].values

        if np.any(vals == 0.0):
            where = np.where(vals == 0.0)
            raise ValueError(
                f"data contains zeros at indexes {where[0].tolist()}, cannot divide by zero"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.loc[:, self.columns].pct_change(periods=1, fill_method=None)

        return result
