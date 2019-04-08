"""Transformers which select either rows of columns."""

from typing import List, Tuple, Type, TypeVar, Union, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .base import BasePandasTransformer


class RowSelector(BasePandasTransformer):
    """A row selector on pandas DataFrames.
    Works on integer indexes using DataFrame.loc method.
    """

    def __init__(self, start=0, end=-1):
        self._check_init_params(start, end)

        self.start = start
        self.end = end

    def fit(self, X: pd.DataFrame, y=None) -> Type['RowSelector']:
        """Select some rows from the dataframe."""
        self._check_types(X)

        n_rows = X.shape[0]
        assert self.start < n_rows, "start cannot be larger than data length"
        assert self.end > -n_rows, f"index end = {self.end} is out of bounds, max backward is {-n_rows}"
        assert self.end <= n_rows, f"index end = {self.end} is out of bounds, max is {n_rows}"

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.start == X.shape[0] - 1 or self.start == -1:
            return X.iloc[[-1], :]  # select last row as a DataFrame!
        else:
            return X.iloc[self.start:self.end, :]

    def _check_init_params(self, start, end):
        if isinstance(start, bool):
            raise TypeError("start cannot be a bool")
        if isinstance(end, bool):
            raise TypeError("end cannot be a bool")

        if not isinstance(start, int):
            raise TypeError(f"start must be an int, got {type(start)}")
        if not isinstance(end, int):
            raise TypeError(f"end must be an int, got {type(end)}")

        if not start >= -1:
            raise ValueError(
                f"start must be greater than or equal to -1, got {start}")

        if start == -1:
            assert end == -1, f"if start is -1, then end should also be -1, got {end}"
        if end > 0:
            assert start < end, "Start must be strictly less than end"


class ColumnSelector(BasePandasTransformer):
    """Select some columns of the data."""

    def __init__(self, columns=Sequence[str]):
        super().__init__(columns)

    def fit(self, X: pd.DataFrame, y=None) -> Type['ColumnSelector']:
        self.prepare_to_fit(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.columns]
