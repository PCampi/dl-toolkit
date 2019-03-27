"""Transformers which operate on a single column."""

from typing import List, Tuple, Type, TypeVar, Union, Callable

import numpy as np
import pandas as pd

from .base import BasePandasTransformer


class LogTransformer(BasePandasTransformer):
    """Apply a logarithm to the data specified in the column."""

    def __init__(self, column: str):
        self._check_init_params(column)

        self.column = column

    def fit(self, X: pd.DataFrame, y=None) -> Type['LogTransformer']:
        self.check_types(X)
        if not np.all(X.loc[:, self.column] > 0):
            raise ValueError(
                "cannot compute the log of zero, there are some zero entries in X"
            )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return np.log(X.loc[:, self.column].values)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.exp(X)

    def _check_init_params(self, column):
        if not isinstance(column, str):
            raise TypeError("the column must be supplied as string")

        if column == '':
            raise ValueError("column must not be the empty string")


class MovingAverageTransformer(BasePandasTransformer):
    """Moving average."""

    def __init__(self, column: str, window=10, kind='simple'):
        self._check_init_params(column, window, kind)

        self.column = column
        self.window = window
        self.kind = kind

    def fit(self, X: pd.DataFrame, y=None) -> Type['MovingAverage']:
        self.check_types(X)

        if self.window >= X.shape[0]:
            raise ValueError(
                f"window ({self.window}) is larger than data length ({X.shape[0]})"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.kind == 'simple':
            rolling = X.loc[:, [self.column]].rolling(
                window=self.window).mean()
        elif self.kind in {'exp', 'exponential'}:
            rolling = X.loc[:, [self.column]].ewm(
                span=self.window,
                adjust=True).mean()  # use adjusted as Yulu suggested

        return rolling

    def _check_init_params(self, column: str, window: int, kind: str):
        self._check_str(column, 'column')

        if not isinstance(window, int):
            raise TypeError(
                f"window parameter must be an integer, got {type(window)}")

        if window <= 1:
            raise ValueError("window must be greater than 1, got {window}")

        if kind not in {'simple', 'exp', 'exponential'}:
            raise ValueError(
                f"kind must be one of 'simple', 'exp' or 'exponential', got {kind}"
            )
