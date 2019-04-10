"""Test file for divide_dataset."""

import numpy as np
import pandas as pd

import src.preprocessing.divide_dataset as dd
import src.preprocessing as opp

if __name__ == "__main__":
    study_len = 1000
    train_len = 750
    bptt = 240

    data = pd.read_csv('./test_data/datesAdjustedPrices.csv'
                       ).loc[:, ['companyId', 'date', 'close']]
    data.loc[:, 'date'] = pd.to_datetime(data.loc[:, 'date'], format="%Y%m%d")
    data.loc[:, 'close'] = data.loc[:, 'close'].astype(np.float32)

    tmp: pd.DataFrame = data.pivot(
        index='date', columns='companyId', values='close').replace(
            -1.0, np.nan)

    # calculate percent returns on the whole dataset
    ts: pd.DataFrame = opp.PercentReturns(tmp.columns).fit_transform(tmp)
    ts = ts.iloc[1:, :]  # delete first row of all NaN

    period = 0
    first_slice: pd.DataFrame = ts.iloc[period * study_len:(period + 1) *
                                        study_len, :]
    first_slice = first_slice.dropna(axis='columns', how='all')

    std_scaler = opp.StandardScaler(first_slice.columns, train_len=train_len)
    std_first_slice = std_scaler.fit_transform(first_slice)

    # try train and test only
    ds_creator = dd.DatasetCreator(
        columns=std_first_slice.columns,
        train_len=train_len,
        bptt=bptt,
        interactive=True)

    X_train, X_test, y_train, y_test = ds_creator.fit_transform(
        std_first_slice)
