"""Test the percentage column difference transformer."""

import numpy as np
import numpy.testing as nt
import pandas as pd
import pandas.testing as pt
import pytest

import src.preprocessing as pp


@pytest.fixture
def data():
    data = {
        'f1': np.array([100, 110, 98, 1500, 30]),
        'f2': 100 * np.ones((5, )),
        'f3': np.zeros((5, )),
        'target1': 100 + np.arange(5),
        'target2': 200 + np.arange(5),
    }

    return pd.DataFrame(data)


def test_log_ret_init():
    with pytest.raises(TypeError):
        pp.LogReturns('1')

    with pytest.raises(TypeError):
        pp.LogReturns(True)

    with pytest.raises(ValueError):
        pp.LogReturns(0)


def test_it_checks_no_zeros_in_a(data):
    with pytest.raises(ValueError):
        pr = pp.PercentReturns()
        pr.fit(data.loc[:, ['f3', 'target1']])

    with pytest.raises(ValueError):
        lr = pp.LogReturns()
        lr.fit(data.loc[:, ['f3', 'target1']])


def test_it_computes_log_returns(data):
    ground_log: pd.DataFrame = np.log(data[['f2', 'target1']])
    ground = ground_log.diff(periods=1)

    lr_transformer = pp.LogReturns()
    result = lr_transformer.fit_transform(data[['f2', 'target1']])

    pt.assert_frame_equal(ground, result)


def test_it_computes_perc_returns(data: pd.DataFrame):
    pr = pp.PercentReturns()
    result = pr.fit_transform(data[['f2', 'f1']])

    data_vals = data[['f2', 'f1']].values

    expected_vals: np.ndarray = np.zeros_like(data_vals)
    expected_vals[0, :] = np.nan  # pylint: disable=unsupported-assignment-operation
    expected_vals[1:] = (data_vals[1:] - data_vals[:-1]) / data_vals[:-1]  # pylint: disable=unsupported-assignment-operation

    expected = pd.DataFrame(
        data=expected_vals,
        columns=data.loc[:, ['f2', 'f1']].columns,
        index=data.loc[:, ['f2', 'f1']].index)

    pt.assert_frame_equal(expected, result)
