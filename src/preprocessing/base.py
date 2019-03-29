"""Transformers module."""

from typing import List, Tuple, Type, TypeVar, Union, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class BasePandasTransformer(BaseEstimator, TransformerMixin):
    """Base class for a pandas transformer.
    Provides facilities for type checking and error reporting.
    """

    def __init__(self, columns: Sequence[str]):
        self.index_ = None
        self.columns = self.check_column_params(columns)

    def check_column_params(self, columns: Sequence[str]):
        """Check that the columns parameter is sane.

        Allowed forms for the column parameter are:
        - list of str
        - tuple of str
        - str

        The output of this function is always a normalized version of the
        column parameter, as a list of str.
        """
        cols = self.__normalize_column_param(columns)

        for i, column in enumerate(cols):
            self._check_str(column, f"column[{i}]")

        return cols

    def __normalize_column_param(self, columns: Sequence[str]):
        if isinstance(columns, list):
            if len(columns) < 1:
                raise ValueError("empty column parameter")
            return columns
        elif isinstance(columns, tuple):
            if len(columns) < 1:
                raise ValueError("empty column parameter")
            return [*columns]
        elif isinstance(columns, str):
            return [columns]
        else:
            raise TypeError(
                f"column parameter must be a Sequence[str], not {type(columns)}"
            )

    def prepare_to_fit(self, X: pd.DataFrame, y=None):
        """Check the type of X (optionally y) and save the index of X."""
        self._check_types(X, y)

        X_cols = X.columns
        for i, column in enumerate(self.columns):
            if not column in X_cols:
                raise ValueError(
                    f"column[{i}] ({column}) not found in input data")

        self.index_ = X.index

    def _check_types(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X should be a {pd.DataFrame}, got {type(X)}")

        if (y is not None) and (not isinstance(y, np.ndarray)):
            raise TypeError(f"y should be a {np.ndarray}, got {type(y)}")

    def _check_str(self, s: str, param_name: str):
        if not isinstance(s, str):
            raise TypeError(f"{param_name} should be a str, not {type(s)}")

        if s == '':
            raise ValueError(f"{param_name} cannot be the empty string")
