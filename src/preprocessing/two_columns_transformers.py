"""Transformers which operate on two columns."""

from typing import List, Tuple, Type, TypeVar, Union, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .base import BasePandasTransformer


class TwoColumnsTransformer(BasePandasTransformer):
    """Applies the supplied transformation to two columns, given by their names."""

    def __init__(self,
                 columns: Sequence[str],
                 operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 new_col_name: str,
                 safety_checks=(None, None)):
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
        new_col_name: str,
            name of the newly created column for the output DataFrame
        safety_checks: tuple of Callable[[np.ndarray], None], optional
            callables that will safety-check column A and B.
            Should raise ValueError if errors, otherwise ok.
            If given, it must be of len = 2.
        """
        if not isinstance(columns, list):
            raise TypeError(
                f"columns must be a list of two str, not {type(columns)}")

        if len(columns) != 2:
            raise ValueError(f"len(columns) must be 2, is {len(columns)}")

        if not isinstance(new_col_name, str):
            raise TypeError(
                f"new_col_name must be a str, not {type(new_col_name)}")
        if new_col_name == '':
            raise ValueError("must give a name to the new column")

        super().__init__(columns)
        self.col_a, self.col_b = columns
        self.new_col_name = new_col_name

        if operation is None:
            raise TypeError("operation cannot be none")
        self.operation = operation

        self.safety_check_a, self.safety_check_b = safety_checks

    def fit(self, X: pd.DataFrame, y=None) -> Type['TwoColumnsTransformer']:
        self.prepare_to_fit(X)

        a = X.loc[:, self.col_a].values
        if self.safety_check_a:
            self.safety_check_a(a)  # pylint: disable=not-callable

        b = X.loc[:, self.col_b].values
        if self.safety_check_b:
            self.safety_check_b(b)  # pylint: disable=not-callable

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        a = X.loc[:, self.col_a].values
        b = X.loc[:, self.col_b].values

        result_values = self.operation(a, b)
        result = pd.DataFrame(
            data=result_values, index=self.index_, columns=[self.new_col_name])

        return result


class PercentChangeTransformer(TwoColumnsTransformer):
    """Given two columns A and B, it computes the
    percentage difference between A and B, with respect to A.

    It means (B - A) / A, or (p_t+1 - p_t) / p_t
    """

    def __init__(self, columns: Sequence[str]):
        def operation(a: np.ndarray, b: np.ndarray):
            return (b - a) / a

        def safety(a: np.ndarray):
            if not np.all(a != 0):
                raise ValueError(
                    "all elements of column A must be greater than 0")

        new_col_name = f"delta_percent_{columns[1]}_{columns[0]}"
        super().__init__(columns, operation, new_col_name, (safety, None))

    def fit(self, X: pd.DataFrame, y=None) -> Type['PercentChangeTransformer']:
        super().fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = super().transform(X)

        return result
