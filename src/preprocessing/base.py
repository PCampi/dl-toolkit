"""Transformers module."""

from typing import List, Tuple, Type, TypeVar, Union, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class EmptyDataFrameError(Exception):
    pass


class BasePandasTransformer(BaseEstimator, TransformerMixin):
    """Base class for a pandas transformer.
    Provides facilities for type checking and error reporting.
    """
    index_: pd.Index
    columns_: pd.Index

    def prepare_to_fit(self, X: pd.DataFrame, y=None):
        """Check the type of X (optionally y) and save the index and columns of X."""
        self._check_types(X, y)

        if X.empty:
            raise EmptyDataFrameError(f"X is empty with shape {X.shape}")

        self.index_ = X.index

        X_cols = X.columns

        # prevent returning a series from 'transform' in subclasses
        if len(X_cols) == 1:
            self.columns_ = X_cols.tolist()
        else:
            self.columns_ = X_cols

    def _check_types(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X should be a {pd.DataFrame}, not {type(X)}")

        if (y is not None) and (not isinstance(y, np.ndarray)):
            raise TypeError(f"y should be a {np.ndarray}, got {type(y)}")

    def _check_str(self, s: str, param_name: str):
        if not isinstance(s, str):
            raise TypeError(f"{param_name} should be a str, not {type(s)}")

        if s == '':
            raise ValueError(f"{param_name} cannot be the empty string")
