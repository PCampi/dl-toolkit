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
