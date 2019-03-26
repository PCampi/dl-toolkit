"""Transformers which operate on two columns."""

from typing import List, Tuple, Type, TypeVar, Union, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .base import BasePandasTransformer


class TwoColumnsTransformer(BasePandasTransformer):
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
        self._check_init_params(col_a, col_b)

        self.col_a = col_a
        self.col_b = col_b
        self.operation = operation

        if safety_check_a:
            self.safety_check_a = safety_check_a
        if safety_check_b:
            self.safety_check_b = safety_check_b

    def fit(self, X: pd.DataFrame, y=None) -> Type['TwoColumnsTransformer']:
        self.check_types(X)
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
        self.check_X_type(X)
        a = X.loc[:, self.col_a].values
        b = X.loc[:, self.col_b].values

        result = self.operation(a, b)

        return result

    def _check_init_params(self, col_a, col_b):
        if not isinstance(col_a, str) or not isinstance(col_b, str):
            raise TypeError(
                f"parameters col_a and col_b should be strings, got {type(col_a)} and {type(col_a)} instead"
            )

        if col_a == '':
            raise ValueError("parameter col_a is empty string")
        if col_b == '':
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
