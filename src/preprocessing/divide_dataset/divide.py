"""Divide a dataset which has a date column."""

from typing import Sequence, Tuple, Type

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..base import BasePandasTransformer


class TimeseriesDateIntervalComputer(BasePandasTransformer):
    """Compute the date intervals to divide a timeseries."""

    def __init__(self, date_column: str, period_len: int, period_shift: int):
        """Divide a timeseries date columns in running windows of len = period_len,
        each shifted by period_shift.

        Parameters
        ----------
        date_column: str
            the name of the column containing dates
        
        period_len: int
            length of the rolling window applied
        
        period_shift: int
            shift between two periods, such that
            (period_i+1).start = period_i.start + period_shift
        """
        if not isinstance(date_column, str):
            raise TypeError(
                f"date_column must be str, not {type(date_column)}")

        super().__init__(date_column)
        super()._check_str(date_column, 'date_column')

        self.date_column = date_column

        if not isinstance(period_len, int):
            raise TypeError(f"period_len must be int, not {type(period_len)}")

        if period_len < 1:
            raise ValueError("period_len must be >= 1")

        if not isinstance(period_shift, int):
            raise TypeError(
                f"period_shift must be int, not {type(period_shift)}")

        if period_shift < 1:
            raise ValueError("period_shift must be >= 1")

        self.period_len = period_len
        self.period_shift = period_shift

    def fit(self, X: pd.DataFrame,
            y=None) -> Type['TimeseriesDateIntervalComputer']:
        self.prepare_to_fit(X)

        date_col_index = X.columns.get_loc(self.date_column)

        if not type(X.iloc[0, date_col_index]) == pd.Timestamp:
            raise TypeError(
                f"column {self.date_column} elements must be of type pd.Timestamp"
            )

        if len(X.columns) < 2:
            raise ValueError(
                "X must have at least two columns (date + values)")

        return self

    def transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        dates = np.sort(X.loc[:, self.date_column].unique())
        n = len(dates)

        start_index = 0
        end_index = 0

        i = 0

        result_dates = []
        result_indexes = []

        while end_index < n - 1:
            i += 1
            end_index = start_index + self.period_len

            if end_index >= n:
                end_index = n - 1

            start_date = dates[start_index]
            end_date = dates[end_index]

            result_dates.append((start_date, end_date))
            result_indexes.append((start_index, end_index))

            start_index = start_index + self.period_shift

        return result_dates, result_indexes
