"""Make a dataset for a study."""

from typing import Sequence, Type

import numpy as np
import pandas as pd
import tqdm

from src import preprocessing as opp


class DatasetCreator(opp.BasePandasTransformer):
    def __init__(self,
                 columns: Sequence[str],
                 train_len=750,
                 bptt=240,
                 train_valid_split=0.2,
                 interactive=True):
        """Make a dataset inside a study period,
        starting from a pd.DataFrame ordered by index.
        
        The passed dataset must be the one considered for a *single* study period.

        Inside a study period, we take a training period of train_len, and
        inside the training period we move from the beginning onwards in blocks
        of window_len days.
        """
        super().__init__(columns)
        self.train_len = train_len
        self.bptt = bptt
        self.train_valid_split = train_valid_split
        self.interactive = interactive

    def fit(self, X: pd.DataFrame, y=None) -> Type['DatasetCreator']:
        """Check dimensions and prepare to fit.

        Most importantly, keep track of the names of the companies
        that were included in the S&P500 at the end of the training period,
        to keep only them in the training and testing phases.

        This is to follow Fischer, Krausse 2018.
        """
        self.prepare_to_fit(X)
        nr, _ = X.shape

        if nr <= self.train_len:
            raise ValueError("there are no days left for testing")

        if nr <= self.bptt:
            raise ValueError("dataset is smaller than bptt")

        last_train_day = X.iloc[self.train_len - 1, :].dropna()
        self.companies_included = last_train_day.index.tolist()

        return self

    def transform(self, X: pd.DataFrame):
        """Get the training and testing data.

        IMPORTANT: the X variable should contain all the data for a
        **single** study period and not be longer, otherwise all indexes get messed up.
        """
        final_data = X.loc[:, self.companies_included]

        # declare the X and y
        X_train, y_train = self._get_slice(final_data, self.bptt, self.bptt)
        if X_train is None or y_train is None:
            raise ValueError("initial X_tarin and y_train are empty!")

        # training period
        # for every slice of `bptt` days, i is the end index
        for_range = range(self.bptt + 1, self.train_len - 1)
        if self.interactive:
            for_range = tqdm.tqdm(for_range)

        for i in for_range:
            tmp_X, tmp_y = self._get_slice(final_data, i, self.bptt)

            X_train = np.concatenate((X_train, tmp_X))
            y_train = np.concatenate((y_train, tmp_y))

        # testing period
        # declare the X and y
        X_test, y_test = self._get_slice(final_data, self.train_len, self.bptt)
        if X_train is None or y_train is None:
            raise ValueError("initial X_tarin and y_train are empty!")
        # for every slice of `bptt` days, i is the end index
        n = final_data.shape[0]

        for_range = range(self.train_len + 1, n)
        if self.interactive:
            for_range = tqdm.tqdm(for_range)

        for i in for_range:
            tmp_X, tmp_y = self._get_slice(final_data, i, self.bptt)

            X_test = np.concatenate((X_test, tmp_X))
            y_test = np.concatenate((y_test, tmp_y))

        return X_train, y_train, X_test, y_test

    def _get_slice(self, data: pd.DataFrame, start_index: int, bptt: int):
        # the first bptt elements of the slice are the X, the last one
        # is the target y
        rolling_slice: pd.DataFrame = data.iloc[start_index -
                                                bptt:start_index + 1, :]
        start_time = rolling_slice.index[0]

        # drop companies that don't have all points in this slice
        rolling_slice = rolling_slice.dropna(axis='columns', how='any')
        r, c = rolling_slice.shape

        if r == 0 or c == 0:
            print(
                f"\n\nWARNING: r={r} and c={c} in slice starting at time={start_time}"
            )
            return None, None

        X = rolling_slice.iloc[:-1, :].values.transpose().reshape(c, r - 1, -1)
        # TODO: fix this using median instead of just y, and assign +1 and 0 classes
        # if the value is above or below the slice median.
        # NOTE: the median is computed on the whole study period, or for a single bptt pass?
        y = rolling_slice.iloc[-1, :].values

        return X, y


def check_args(data, train_dates, test_dates, rolling_window):
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be pd.DataFrame, not {type(data)}")

    if not all(isinstance(date, np.datetime64) for date in train_dates):
        raise TypeError("all training dates must be pd.Timestamp")

    if not all(isinstance(date, np.datetime64) for date in test_dates):
        raise TypeError("all test dates must be pd.Timestamp")

    if not isinstance(rolling_window, int):
        raise TypeError(
            f"rolling window must be int, not {type(rolling_window)}")

    if rolling_window < 1:
        raise ValueError("rolling window must be >= 1")


def check_has_columns(data: pd.DataFrame):
    if not 'date' in data.columns:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(
                "no column named 'date' in the data and the index is not a datetime"
            )
