"""Test the column selector."""

import pytest
import numpy as np
import pandas as pd
import numpy.testing as nt
import pandas.testing as pt
from sklearn.exceptions import NotFittedError

import preprocessing as pp


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
    with pytest.raises(AssertionError):
        pp.PercentChangeTransformer(True, 'age')

    with pytest.raises(AssertionError):
        pp.PercentChangeTransformer(0, 1.4)

    with pytest.raises(AssertionError):
        pp.PercentChangeTransformer(1, -1)
    with pytest.raises(AssertionError):
        pp.PercentChangeTransformer(-2, 1)


def test_it_checks_columns_in_df(data: pd.DataFrame):
    with pytest.raises(AssertionError):
        pt = pp.PercentChangeTransformer('f1', 'target3')
        pt.fit(data)

    with pytest.raises(AssertionError):
        pt = pp.PercentChangeTransformer('target3', 'f1')
        pt.fit(data)

    l = len(data.columns)
    with pytest.raises(AssertionError):
        pt = pp.PercentChangeTransformer('f1', l)
        pt.fit(data)

    with pytest.raises(AssertionError):
        pt = pp.PercentChangeTransformer(l, 'f1')
        pt.fit(data)

    with pytest.raises(AssertionError):
        pt = pp.PercentChangeTransformer('f1', -1)
        pt.fit(data)

    with pytest.raises(AssertionError):
        pt = pp.PercentChangeTransformer(-1, 'f1')


def test_it_checks_no_zeros_in_a(data):
    with pytest.raises(AssertionError):
        pt = pp.PercentChangeTransformer('f3', 'target1')
        pt.fit(data)


def test_it_transforms_data(data: pd.DataFrame):
    pt = pp.PercentChangeTransformer('f2', 'f1')
    result = pt.fit_transform(data)

    expected = np.array([0.0, 0.1, -0.02, 14.0, -0.7])

    nt.assert_array_equal(result, expected)
