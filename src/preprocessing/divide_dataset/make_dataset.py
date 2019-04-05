"""Make a dataset for a study."""

from typing import List, Set

import numpy as np
import pandas as pd
import sklearn.pipeline as skpp

import tqdm

from src.rnn_input_creator import create_dataset

from ..price_transformation import PercentReturns
from ..scaling import StandardScaler


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


def get_slice(data: pd.DataFrame, start_index: int, bptt: int):
    # the first bptt elements of the slice are the X, the last one
    # is the target y
    rolling_slice: pd.DataFrame = data.iloc[start_index - bptt:start_index +
                                            1, :]
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
    y = rolling_slice.iloc[-1, :].values

    return X, y


def get_training_dataset(train_data: pd.DataFrame, bptt: int):
    """Move inside the training data using windows of len = bptt,
    and for each window take the series for every stock that was present
    in all days of that window.

    Assume that we are trying to predict the next value in each column.
    """
    train_len = len(train_data)
    if bptt >= train_len:
        raise ValueError(f"bptt ({bptt}) > len(train_data) ({train_len})")

    # declare the X and y
    i = bptt
    X, y = get_slice(train_data, i, bptt)
    if X is None or y is None:
        raise ValueError("initial X and y are empty!")

    # for every slice of `rolling_window` days, i is the end index
    for i in tqdm.tqdm(range(bptt + 1, train_len - 1)):
        tmp_X, tmp_y = get_slice(train_data, i, bptt)

        X = np.concatenate((X, tmp_X))
        y = np.concatenate((y, tmp_y))

    return X, y


def make_ds_in_study_period(data: pd.DataFrame, train, test, bptt):
    """Make a dataset for a study period defined by the training and test dates.
    
    Advance in time by slices of len = rolling_window.
    
    Data must contain a column named 'date', with dtype pd.Timestamp,
    and a column named 'companyId', of str.
    """
    check_has_columns(data)

    # 0. get dates and indexes
    train_start_date, train_end_date = train['dates']
    test_start_date, test_end_date = test['dates']

    check_args(data, (train_start_date, train_end_date),
               (test_start_date, test_end_date), bptt)

    train_len = train['indexes'][1] - train['indexes'][0]

    # 0b. get the data subset for the whole study period
    subset_study: pd.DataFrame = data[(data.index >= train_start_date)
                                      & (data.index <= test_end_date)]

    subset_study = subset_study.dropna(axis='columns', how='all')

    # 1. compute returns and standardize
    returns_transformer = PercentReturns(columns=subset_study.columns)
    standard_scaler = StandardScaler(columns=subset_study.columns)

    tr = skpp.Pipeline(
        steps=[('returns',
                returns_transformer), ('standardize', standard_scaler)])

    std_returns = tr.fit_transform(subset_study)
    # drop the first NaN otherwise it breaks everything
    std_returns = std_returns.drop(labels=std_returns.index[0], axis='index')

    # 2. iterate over the rolling_window slices
    n_rows_study = subset_study.shape[0]
    assert n_rows_study >= bptt + train_len, "there is no data left for testing"

    X, y = get_training_dataset(std_returns.iloc[:train_len, :], bptt)

    return X, y
