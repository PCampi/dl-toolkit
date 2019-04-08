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


def test_it_checks_init_params(data: pd.DataFrame):
    with pytest.raises(TypeError):
        pp.PercentReturns((True, 'age'))

    with pytest.raises(TypeError):
        pp.PercentReturns((0, 1.4))


def test_it_checks_columns_in_df(data: pd.DataFrame):
    with pytest.raises(ValueError):
        pt = pp.PercentReturns(['f1', 'target3'])
        pt.fit(data)

    with pytest.raises(ValueError):
        pt = pp.PercentReturns(['target3', 'f1'])
        pt.fit(data)


def test_it_checks_no_zeros_in_a(data):
    with pytest.raises(ValueError):
        pt = pp.PercentReturns(['f3', 'target1'])
        pt.fit(data)


def test_it_transforms_data(data: pd.DataFrame):
    pr = pp.PercentReturns(['f2', 'f1'])
    result = pr.fit_transform(data)

    data_vals = data.loc[:, ['f2', 'f1']].values

    expected_vals: np.ndarray = np.zeros_like(data_vals)
    expected_vals[0, :] = np.nan  # pylint: disable=unsupported-assignment-operation
    expected_vals[1:] = (data_vals[1:] - data_vals[:-1]) / data_vals[:-1]  # pylint: disable=unsupported-assignment-operation

    expected = pd.DataFrame(
        data=expected_vals,
        columns=data.loc[:, ['f2', 'f1']].columns,
        index=data.loc[:, ['f2', 'f1']].index)

    pt.assert_frame_equal(expected, result)
