"""Test file for divide_dataset."""

import numpy as np
import pandas as pd

import src.preprocessing.divide_dataset as dd

if __name__ == "__main__":
    train_len = 250
    test_len = 250
    bptt = 25

    data = pd.read_csv('./test_data/datesAdjustedPrices.csv')

    data = data.drop(columns=['indexId', 'open', 'high', 'low'])
    data.loc[:, 'date'] = pd.to_datetime(data.loc[:, 'date'], format="%Y%m%d")

    date_divider = dd.TimeseriesDateIntervalComputer(
        'date',
        period_len=train_len + test_len,
        period_shift=test_len,
        train_len=train_len)

    all_dates, train_periods, test_periods = date_divider.fit_transform(data)

    ts: pd.DataFrame = data.pivot(
        index='date', columns='companyId', values='close').replace(
            -1.0, np.nan)

    Xs = []
    ys = []
    n = len(train_periods)

    for i, (train_period, test_period) in enumerate(
            zip(train_periods, test_periods)):
        print(f"\nPeriod {i + 1}")
        X, y = dd.make_ds_in_study_period(ts, train_period, test_period, bptt)
        Xs.append(X)
        ys.append(y)

    X_train = np.concatenate(Xs)
    y_train = np.concatenate(ys)
