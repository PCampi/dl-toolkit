"""Transformer which wraps the specified data into a DataFrame, with the specified columns."""

from typing import Type, Sequence

import numpy as np
import pandas as pd
import sklearn.preprocessing as skpp
from sklearn.base import TransformerMixin

from .base import BasePandasTransformer


class BaseScaler(BasePandasTransformer):
    """Base class for pandas scalers."""

    def __init__(self, columns: Sequence[str]):
        super().__init__(columns)
        self.scaler_: TransformerMixin = None

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed_data = self.scaler_.transform(
            X.loc[:, self.columns].values)

        result = pd.DataFrame(
            data=transformed_data, columns=self.columns, index=self.index_)

        return result

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        inv_transformed_data = self.scaler_.inverse_transform(
            X.loc[:, self.columns].values)

        result = pd.DataFrame(
            data=inv_transformed_data, columns=self.columns, index=self.index_)

        return result


class MinMaxScaler(BaseScaler):
    """Scale the input in a range."""

    def __init__(self, columns: Sequence[str], feature_range=(0, 1)):
        super().__init__(columns)

        self.feature_range = feature_range

    def fit(self, X: pd.DataFrame, y=None) -> Type['MinMaxScaler']:
        self.prepare_to_fit(X)
        self.scaler_ = skpp.MinMaxScaler(feature_range=self.feature_range)

        self.scaler_.fit(X.loc[:, self.columns].values)

        return self


class StandardScaler(BaseScaler):
    """Scale the input in a range."""

    def __init__(self,
                 columns: Sequence[str],
                 copy=True,
                 with_mean=True,
                 with_std=True):
        super().__init__(columns)

        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X: pd.DataFrame, y=None) -> Type['MinMaxScaler']:
        self.prepare_to_fit(X)
        self.scaler_ = skpp.StandardScaler(
            copy=self.copy, with_mean=self.with_mean, with_std=self.with_std)

        self.scaler_.fit(X.loc[:, self.columns].values)

        return self
