"""Transformer which wraps the specified data into a DataFrame, with the specified columns."""

from typing import Optional, Type

import numpy as np
import pandas as pd
import sklearn.preprocessing as skpp
from sklearn.base import TransformerMixin

from .base import BasePandasTransformer


class BaseScaler(BasePandasTransformer):
    """Base class for pandas scalers."""
    scaler_: TransformerMixin = None
    train_len: Optional[int]

    def __init__(self, train_len: int = None):
        if train_len is not None:
            if not isinstance(train_len, int):
                raise TypeError(
                    f"train_len must be int, not {type(train_len)}")

            if train_len < 1:
                raise ValueError(f"invalid training length {train_len}")

        self.train_len = train_len

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed_data = self.scaler_.transform(X.values)

        result = pd.DataFrame(
            data=transformed_data, columns=self.columns_, index=self.index_)

        return result

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        inv_transformed_data = self.scaler_.inverse_transform(X.values)

        result = pd.DataFrame(
            data=inv_transformed_data,
            columns=self.columns_,
            index=self.index_)

        return result


class MinMaxScaler(BaseScaler):
    """Scale the input in a range."""

    def __init__(self, feature_range=(0, 1), copy=True, train_len: int = None):
        super().__init__(train_len)

        self.feature_range = feature_range
        self.copy = copy

    def fit(self, X: pd.DataFrame, y=None) -> Type['MinMaxScaler']:
        """Fit the data.

        X shall be the whole (train + test) dataset, otherwise
        all indexing gets messed up.
        """
        self.prepare_to_fit(X)
        self.scaler_ = skpp.MinMaxScaler(
            feature_range=self.feature_range, copy=self.copy)

        if self.train_len:
            self.scaler_.fit(X.iloc[:self.train_len, :].values)
        else:
            self.scaler_.fit(X.values)

        return self


class StandardScaler(BaseScaler):
    """Scale the input to have zero mean and unit variance."""

    def __init__(self,
                 copy=True,
                 with_mean=True,
                 with_std=True,
                 train_len: int = None):
        super().__init__(train_len)

        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X: pd.DataFrame, y=None) -> Type['MinMaxScaler']:
        """Fit the data.

        X shall be the whole (train + test) dataset, otherwise
        all indexing gets messed up.
        """
        self.prepare_to_fit(X)
        self.scaler_ = skpp.StandardScaler(
            copy=self.copy, with_mean=self.with_mean, with_std=self.with_std)

        if self.train_len:
            self.scaler_.fit(X.iloc[:self.train_len, :].values)
        else:
            self.scaler_.fit(X.values)

        return self
