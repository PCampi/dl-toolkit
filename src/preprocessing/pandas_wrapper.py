"""Transformer which wraps the specified data into a DataFrame, with the specified columns."""

from typing import List, Tuple, Type, TypeVar, Union, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .base import BasePandasTransformer


class DataFrameWrapper(BasePandasTransformer):
    """Take some data and put it into a DataFrame."""

    def __init__(self, columns: List[str]):
        self._check_init_params(columns)
        self.columns = columns

    def fit(self, X: np.ndarray, y=None) -> Type['DataFrameWrapper']:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a np.ndarray, got {type(X)}")

        if X.ndim == 1:
            raise ValueError(
                f"X must have 2 dimensions, got array with shape {X.shape}")

        if not X.shape[1] == len(self.columns):
            raise ValueError(
                f"declared columns length and data.shape[1] do not match ([{len(self.columns)}] / [{X.shape[1]}])"
            )

        return self

    def transform(self, X: np.ndarray) -> pd.DataFrame:
        result = pd.DataFrame(data=X, columns=self.columns)

        return result

    def _check_init_params(self, columns: List[str]):
        if not isinstance(columns, list):
            raise TypeError(f"columns should be a list, got {type(columns)}")

        if len(columns) == 0:
            raise ValueError("columns parameter cannot be the empty list")

        for i, column in enumerate(columns):
            self._check_str(column, f"columns[{i}]")
