"""Transformers which select either rows of columns."""

from typing import List, Tuple, Type, TypeVar, Union, Callable

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
        self.check_types(X)

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
        assert isinstance(start,
                          int), f"Start must be an int, got {type(start)}"
        assert isinstance(end, int), f"End must be an int, got {type(end)}"
        assert start >= -1, f"start must be greater than or equal to -1, got {start}"

        if start == -1:
            assert end == -1, f"if start is -1, then end should also be -1, got {end}"
        if end > 0:
            assert start < end, "Start must be strictly less than end"


class ColumnSelector(BasePandasTransformer):
    """Select some columns of the data."""

    def __init__(self, columns=List[str]):
        self._check_init_params(columns)
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None) -> Type['ColumnSelector']:
        assert all(c in X.columns
                   for c in self.columns), "not all columns are in the data!"
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.columns]

    def _check_init_params(self, columns):
        assert isinstance(
            columns,
            list), f"columns should be a list of str, got {type(columns)}"
        assert len(columns) >= 1, "at least a column name should be provided"
        assert all(isinstance(c, str)
                   for c in columns), "some columns entry is not a str"
        assert all(c != '' for c in columns)
