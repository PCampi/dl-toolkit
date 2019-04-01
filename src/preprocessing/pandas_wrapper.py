"""Transformer which wraps the specified data into a DataFrame, with the specified columns."""

from typing import Callable, List, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one
from sklearn.utils._joblib import Parallel, delayed

from .base import BasePandasTransformer


class SparseNotAllowedError(Exception):
    """Raised when a sparse result is not allowed in the
    processing pipeline.
    """
    pass


class DataFrameWrapper(BasePandasTransformer):
    """Take some data and put it into a DataFrame."""

    def __init__(self, columns: List[str]):
        super().__init__(columns)

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


class PandasFeatureUnion(FeatureUnion):
    """Scikit-learn feature union with support for pandas DataFrames."""

    def merge_dataframes_by_column(self, Xs) -> pd.DataFrame:
        """Merge dataframes stacking them column by column."""
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Input data to be transformed.
        Returns
        -------
        X_t : pd.DataFrame, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter())

        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        if any(sparse.issparse(f) for f in Xs):
            raise SparseNotAllowedError(
                "sparse results are not allowed, check transformers")
        else:
            if not all(isinstance(x, pd.DataFrame) for x in Xs):
                raise TypeError(
                    "one of the results is not a DataFrame, check your transformers"
                )
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def fit_transform(self, X: pd.DataFrame, y=None,
                      **fit_params) -> pd.DataFrame:
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        X_t : pd.DataFrame, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, X, y, weight, **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            raise SparseNotAllowedError(
                "sparse results are not allowed, check transformers")
        else:
            if not all(isinstance(x, pd.DataFrame) for x in Xs):
                raise TypeError(
                    "one of the results is not a DataFrame, check your transformers"
                )
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs
