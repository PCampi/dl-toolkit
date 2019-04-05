"""Transformers which operate on a single column."""

from typing import List, Tuple, Type, TypeVar, Union, Callable, Sequence

import numpy as np
import pandas as pd

from .base import BasePandasTransformer


class LogTransformer(BasePandasTransformer):
    """Apply a logarithm to the data specified in the column."""

    def __init__(self, columns: Sequence[str]):
        super().__init__(columns)

    def fit(self, X: pd.DataFrame, y=None) -> Type['LogTransformer']:
        self.prepare_to_fit(X)

        if not np.all(X.loc[:, self.columns] > 0):
            raise ValueError(
                "cannot compute the log of zero, there are some zero entries in X"
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return np.log(X.loc[:, self.columns])

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return np.exp(X.loc[:, self.columns])


class Log10Transformer(BasePandasTransformer):
    """Apply base-10 logarithm to a column."""

    def fit(self, X: pd.DataFrame, y=None):
        self.prepare_to_fit(X)

        if not np.all(X.loc[:, self.columns] > 0):
            raise ValueError(
                "cannot compute the log of zero, there are some zero entries in X"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return np.log10(X.loc[:, self.columns])

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return np.power(10, X.loc[:, self.columns])


class PercentChangeTransformer(BasePandasTransformer):
    """Percent change over columns, starting from the first element."""

    def __init__(self, columns: Sequence[str], periods=1):
        """Percent change over columns, using periods steps between data points.

        Parameters
        ----------
        column: Sequence[str]
            the columns on which to compute the change
        periods: int
            number of periods between data points to use for computing the change
        """
        if isinstance(periods, bool):
            raise TypeError("cannot use bool as periods")
        if not isinstance(periods, int):
            raise TypeError(f"periods must be int, not {type(periods)}")

        if periods < 1:
            raise ValueError(f"periods must be >= 1, got {periods}")

        super().__init__(columns)
        self.periods = periods

    def fit(self, X: pd.DataFrame, y=None) -> Type['PercentChangeTransformer']:
        self.prepare_to_fit(X)

        if not np.all(X.loc[:, self.columns].values > 0.0):
            raise ValueError("cannot divide by zero and data contains zeros")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.loc[:, self.columns].pct_change(
            periods=self.periods, fill_method=None)

        new_cols = [f"{old_col}_perc_change" for old_col in result.columns]
        result.columns = new_cols

        return result


class MovingAverageTransformer(BasePandasTransformer):
    """Moving average."""

    def __init__(self, column: str, window=10, kind='simple'):
        super().__init__(column)
        self._check_init_params(column, window, kind)

        self.window = window
        self.kind = kind

    def fit(self, X: pd.DataFrame, y=None) -> Type['MovingAverageTransformer']:
        self.prepare_to_fit(X)

        if self.window >= X.shape[0]:
            raise ValueError(
                f"window ({self.window}) is larger than data length ({X.shape[0]})"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.kind == 'simple':
            rolling = X.loc[:, self.columns].rolling(window=self.window).mean()
        elif self.kind in {'exp', 'exponential'}:
            rolling = X.loc[:, self.columns].ewm(
                span=self.window,
                adjust=True).mean()  # use adjusted as Yulu suggested

        return rolling

    def _check_init_params(self, column: str, window: int, kind: str):
        if not isinstance(window, int):
            raise TypeError(
                f"window parameter must be an integer, got {type(window)}")

        if window <= 1:
            raise ValueError("window must be greater than 1, got {window}")

        if kind not in {'simple', 'exp', 'exponential'}:
            raise ValueError(
                f"kind must be one of 'simple', 'exp' or 'exponential', got {kind}"
            )
