"""Transformers module."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class BasePandasTransformer(ABC):
    """Base class for a pandas transformer.
    Provides facilities for type checking and error reporting.
    """

    def check_X_type(self, X, desired_type=pd.DataFrame):
        if not isinstance(X, desired_type):
            raise TypeError(f"X should be a {desired_type}, got {type(X)}")

    def check_y_type(self, y, desired_type=np.ndarray):
        if not isinstance(y, desired_type):
            raise TypeError(f"y should be a {desired_type}, got {type(y)}")


class RowSelector(BaseEstimator, TransformerMixin, BasePandasTransformer):
    """A row selector on pandas DataFrames.
    Works on integer indexes.
    """

    def __init__(self, start=0, end=-1):
        self._check_init_params(start, end)

        self.start_ = start
        self.end_ = end

    def fit(self, X: pd.DataFrame, y=None) -> Type['RowSelector']:
        """Select some rows from the dataframe."""
        self.check_X_type(X)

        n_rows = X.shape[0]
        assert self.start_ < n_rows, "start cannot be larger than data length"
        assert self.end_ > -n_rows, f"index end = {self.end_} is out of bounds, max backward is {-n_rows}"
        assert self.end_ <= n_rows, f"index end = {self.end_} is out of bounds, max is {n_rows}"

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.start_ == X.shape[0] - 1 or self.start_ == -1:
            return X.iloc[[-1], :]  # select last row as a DataFrame!
        else:
            return X.iloc[self.start_:self.end_, :]

    def _check_init_params(self, start, end):
        assert isinstance(start,
                          int), f"Start must be an int, got {type(start)}"
        assert isinstance(end, int), f"End must be an int, got {type(end)}"
        assert start >= -1, f"start must be greater than or equal to -1, got {start}"

        if start == -1:
            assert end == -1, f"if start is -1, then end should also be -1, got {end}"
        if end > 0:
            assert start < end, "Start must be strictly less than end"


class ColumnSelector(BaseEstimator, TransformerMixin, BasePandasTransformer):
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


class PercentChangeTransformer(BaseEstimator, TransformerMixin,
                               BasePandasTransformer):
    """Given two columns A and B, it computes the
    percentage difference between A and B, with respect to A.

    It means (B - A) / A, or (p_t+1 - p_t) / p_t
    """

    def __init__(self, col_a: Union[str, int], col_b: Union[str, int]):
        self.__check_init_params(col_a, col_b)

        self.col_a = col_a
        self.col_b = col_b

    def fit(self, X: pd.DataFrame, y=None):
        self._check_columns(X)

        a = self._select_column(X, self.col_a)

        assert np.all(
            a != 0.0), "cannot use a column with a zero entry as divisor"
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        a = self._select_column(X, self.col_a)
        b = self._select_column(X, self.col_b)

        perc_change = (b - a) / a

        return perc_change

    def _check_columns(self, X: pd.DataFrame):
        self._check_column(X, self.col_a)
        self._check_column(X, self.col_b)

    def _check_column(self, X: pd.DataFrame, column: Union[str, int]):
        if isinstance(column, str):
            assert column in X.columns, "column {column} is not in the dataframe"
        elif isinstance(column, int):
            assert column < len(
                X.columns
            ), f"column number {column} is out of bounds, there are only {len(X.columns)} columns"

    def _select_column(self, X: pd.DataFrame,
                       column: Union[str, int]) -> np.ndarray:
        if isinstance(column, str):
            return X.loc[:, column].values
        elif isinstance(column, int):
            return X.iloc[:, column].values

    def __check_init_params(self, col_a, col_b):
        assert isinstance(col_a, str) or \
            isinstance(col_a, int) and \
            not isinstance(col_a, bool), f"col_a must be a string or integer, got {type(col_a)}"
        assert isinstance(col_b, str) or \
            isinstance(col_b, int) and \
            not isinstance(col_b, bool), f"col_a must be a string or integer, got {type(col_b)}"

        if isinstance(col_a, str):
            assert col_a != '', "column A name is empty string"
        if isinstance(col_b, str):
            assert col_b != '', "column B name is empty string"

        if isinstance(col_a, int):
            assert col_a >= 0, "integer index for column a must be >= 0"
        if isinstance(col_b, int):
            assert col_b >= 0, "integer index for column b must be >= 0"
