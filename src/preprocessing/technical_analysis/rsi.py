"""Transformers module."""

from typing import Type, Sequence

import numpy as np
import pandas as pd

from ..base import BasePandasTransformer


class RsiTransformer(BasePandasTransformer):
    """Compute the Relative Strength Index of a series."""

    def __init__(self, column: str, window=14):
        """Compute the RSI of a stock o index.

        Parameters
        ----------
        columns: the close column in the data
            the names of the column that represent the close price
        
        window: int
            the period considered
        """
        if not isinstance(column, str):
            raise TypeError(f"columns must be str, not {type(column)}")

        if not isinstance(window, int):
            raise TypeError(
                f"initial_period must be a positive int, not {type(window)}")
        if window < 1:
            raise ValueError(f"initial_period must be >= 1, got {window}")

        super().__init__(column)
        self.window = window

    def fit(self, X: pd.DataFrame, y=None) -> Type['RsiTransformer']:
        self.prepare_to_fit(X)
        if self.window > X.shape[0]:
            raise ValueError(
                f"window is larger than data: w={self.window}, d={X.shape[0]}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get the RSI from the data, based on the formula at
        `Wikipedia <https://en.wikipedia.org/wiki/Relative_strength_index>`_

        Parameters
        ----------
        X: pd.DataFrame
            a dataframe containing the two columns defined as open and close prices
        
        Returns
        -------
        pd.DataFrame
            a dataframe with the same index as X, with a single column named RSI
        """
        # 1. compute difference of the closing prices
        close_diff = X.loc[:, self.columns].diff(
            periods=1).values  # 1D array, first is Nan

        nrows = close_diff.shape[0]

        # 2. init the result array and set the first `window` entries to np.nan
        rsi = np.zeros((len(close_diff, )))
        rsi[:self.window] = np.nan

        # 3. initial phase
        initial_period = close_diff[1:self.window + 1]
        assert len(initial_period) == self.window

        avg_gain = np.sum(initial_period[initial_period > 0.0]) / self.window
        avg_loss = np.abs(np.sum(
            initial_period[initial_period < 0.0])) / self.window

        rs = avg_gain / avg_loss if avg_loss != 0.0 else np.nan
        initial_rsi = 100 - 100 / (1 + rs) if not np.isnan(rs) else 100

        rsi[self.window] = initial_rsi

        # 4. rolling calculation for the rest of the data
        for i in range(self.window + 1, nrows):
            curr_delta = close_diff[i]

            curr_gain = curr_delta if curr_delta > 0.0 else 0.0
            curr_loss = np.abs(curr_delta) if curr_delta < 0.0 else 0.0

            avg_gain = (avg_gain * (self.window - 1) + curr_gain) / self.window
            avg_loss = (avg_loss * (self.window - 1) + curr_loss) / self.window

            rs = avg_gain / avg_loss if avg_loss != 0.0 else np.nan
            rsi[i] = 100 - 100 / (1 + rs) if not np.isnan(rs) else 100

        assert not np.isnan(rsi[-1]), "there is an error in the indexes"

        return pd.DataFrame(data=rsi, columns=['rsi'])
