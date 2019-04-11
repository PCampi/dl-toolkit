"""Transformers which operate on a single column."""

from typing import List, Tuple, Type, TypeVar, Union, Callable, Sequence

import numpy as np
import pandas as pd

from .base import BasePandasTransformer
from .column_transformers import PercentChangeTransformer


class LogReturns(BasePandasTransformer):
    """Take the log return."""

    def __init__(self, periods=1):
        if isinstance(periods, bool):
            raise TypeError("periods must be int, not bool")

        if not isinstance(periods, int):
            raise TypeError(f"periods must be int, not {type(periods)}")

        if periods < 1:
            raise ValueError("periods must be >= 1")

        self.periods = periods

    def fit(self, X: pd.DataFrame, y=None) -> Type['LogReturns']:
        self.prepare_to_fit(X)

        if self.periods >= X.shape[0]:
            raise ValueError(
                f"not enough rows ({X.shape[0]}) for a period of ({self.periods})"
            )

        vals = X.values

        if np.any(vals == 0.0):
            where = np.where(vals == 0.0)
            raise ValueError(
                f"data contains zeros at indexes {where[0].tolist()}, cannot divide by zero"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = np.log(X).diff(periods=1)

        return result


class PercentReturns(PercentChangeTransformer):
    """Take the percentage returns of each column."""

    def __init__(self):
        super().__init__(periods=1)

    def fit(self, X: pd.DataFrame, y=None) -> Type['PercentReturns']:
        super().fit(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return super().transform(X)
