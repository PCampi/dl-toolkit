"""Transformers module."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Type, TypeVar, Union, Callable

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


class TwoColumnsTransformer(BaseEstimator, TransformerMixin,
                            BasePandasTransformer):
    """Applies the supplied transformation to two columns, given by their names."""

    def __init__(self,
                 col_a: str,
                 col_b: str,
                 operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 safety_check_a: Callable[[np.ndarray], None] = None,
                 safety_check_b: Callable[[np.ndarray], None] = None):
        """Creates a new instance of this class.

        Parameters
        ---------- 
        col_a: str
            the name of the first column
        col_b: str 
            the name of the second column
        operation: Callable[[np.ndarray, np.ndarray], np.ndarray]
            the operation to be applied to the two columns. It must return a numpy array
            of the same length
        safety_check_a: Callable[[np.ndarray], None], optional
            callable that will safety-check column A. Raises ValueError if errors, otherwise ok
        safety_check_b: Callable[[np.ndarray], None], optional
            callable that will safety-check column A. Raises ValueError if errors, otherwise ok
        """
        self.__check_init_params(col_a, col_b)

        self.col_a = col_a
        self.col_b = col_b
        self.operation = operation

        if safety_check_a:
            self.safety_check_a = safety_check_a
        if safety_check_b:
            self.safety_check_b = safety_check_b

    def fit(self, X: pd.DataFrame, y=None):
        self.__check_columns(X)

        a = X.loc[:, self.col_a].values
        try:
            self.safety_check_a(a)
        except AttributeError:
            pass

        b = X.loc[:, self.col_b].values
        try:
            self.safety_check_b(b)
        except AttributeError:
            pass

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        a = X.loc[:, self.col_a].values
        b = X.loc[:, self.col_b].values

        result = self.operation(a, b)

        return result

    def __check_init_params(self, col_a, col_b):
        if not isinstance(col_a, str) or not isinstance(col_b, str):
            raise TypeError(
                f"parameters col_a and col_b should be strings, got {type(col_a)} and {type(col_a)} instead"
            )

        if not col_a != '':
            raise ValueError("parameter col_a is empty string")
        if not col_b != '':
            raise ValueError("parameter col_b is empty string")

    def __check_columns(self, X: pd.DataFrame):
        self.__check_column(X, self.col_a)
        self.__check_column(X, self.col_b)

    def __check_column(self, X: pd.DataFrame, column: str):
        if not column in X.columns:
            raise ValueError(f"column {column} is not in the dataframe")


class PercentChangeTransformer(TwoColumnsTransformer):
    """Given two columns A and B, it computes the
    percentage difference between A and B, with respect to A.

    It means (B - A) / A, or (p_t+1 - p_t) / p_t
    """

    def __init__(self, col_a: str, col_b: str):
        def operation(a: np.ndarray, b: np.ndarray):
            return (b - a) / a

        def safety(a: np.ndarray):
            if not np.all(a != 0):
                raise ValueError(
                    "all elements of column A must be greater than 0")

        super().__init__(col_a, col_b, operation, safety_check_a=safety)

    def fit(self, X: pd.DataFrame, y=None):
        super().fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        result = super().transform(X)

        return result
