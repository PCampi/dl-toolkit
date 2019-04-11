"""Divide a dataset which has a date column."""

from typing import Sequence, Tuple, Type

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..base import BasePandasTransformer


class TimeseriesDateIntervalComputer(BasePandasTransformer):
    """Compute the date intervals to divide a timeseries."""

    def __init__(self, date_column: str, period_len: int, period_shift: int,
                 train_len: int):
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
        
        train_len: int
            length of the training part inside a study period. Must be strictly
            less than period_len
        """
        if not isinstance(date_column, str):
            raise TypeError(
                f"date_column must be str, not {type(date_column)}")

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

        if not isinstance(train_len, int):
            raise TypeError(f"period_len must be int, not {type(train_len)}")
        if train_len >= period_len:
            raise ValueError(
                f"train len ({train_len}) must be strictly less than period_len ({period_len})"
            )

        self.period_len = period_len
        self.period_shift = period_shift
        self.train_len = train_len

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

    def transform(self, X: pd.DataFrame
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the intervals and the indexes of the study periods
        in the date column.
        """
        dates = np.sort(X.loc[:, self.date_column].unique())
        n = len(dates)

        study_start_index = 0
        study_end_index = 0

        train_start_index = 0
        train_end_index = 0
        test_start_index = 0
        test_end_index = 0

        i = 0

        train_intervals = []
        test_intervals = []

        while study_end_index < n - 1:
            i += 1
            study_end_index = study_start_index + self.period_len

            if study_end_index >= n:
                study_end_index = n - 1

            train_end_index = train_start_index + self.train_len
            test_start_index = train_end_index + 1
            test_end_index = study_end_index

            train_start_date = dates[train_start_index]
            train_end_date = dates[train_end_index]

            test_start_date = dates[test_start_index]
            test_end_date = dates[test_end_index]

            train = {
                'dates': (train_start_date, train_end_date),
                'indexes': (train_start_index, train_end_index),
            }

            test = {
                'dates': (test_start_date, test_end_date),
                'indexes': (test_start_index, test_end_index)
            }

            train_intervals.append(train)
            test_intervals.append(test)

            study_start_index = study_start_index + self.period_shift
            train_start_index = study_start_index

        return dates, train_intervals, test_intervals
