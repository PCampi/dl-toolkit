"""Transformers module."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Type, TypeVar, Union, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BasePandasTransformer(BaseEstimator, TransformerMixin):
    """Base class for a pandas transformer.
    Provides facilities for type checking and error reporting.
    """

    def check_X_type(self, X, desired_type=pd.DataFrame):
        if not isinstance(X, desired_type):
            raise TypeError(f"X should be a {desired_type}, got {type(X)}")

    def check_y_type(self, y, desired_type=np.ndarray):
        if y is None:
            return

        if not isinstance(y, desired_type):
            raise TypeError(f"y should be a {desired_type}, got {type(y)}")

    def check_types(self, X, y=None, X_type=pd.DataFrame, y_type=None):
        self.check_X_type(X, X_type)
        self.check_y_type(y, y_type)

    def _check_str(self, s: str, param_name: str):
        if not isinstance(s, str):
            raise TypeError(f"{param_name} should be a str, got {type(s)}")

        if s == '':
            raise ValueError(f"{param_name} cannot be the empty string")

    def _check_init_params(self, *args):
        raise NotImplementedError
