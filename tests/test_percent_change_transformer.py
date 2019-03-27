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
        pp.PercentChangeTransformer(True, 'age')

    with pytest.raises(TypeError):
        pp.PercentChangeTransformer(0, 1.4)


def test_it_checks_columns_in_df(data: pd.DataFrame):
    with pytest.raises(ValueError):
        pt = pp.PercentChangeTransformer('f1', 'target3')
        pt.fit(data)

    with pytest.raises(ValueError):
        pt = pp.PercentChangeTransformer('target3', 'f1')
        pt.fit(data)


def test_it_checks_no_zeros_in_a(data):
    with pytest.raises(ValueError):
        pt = pp.PercentChangeTransformer('f3', 'target1')
        pt.fit(data)


def test_it_transforms_data(data: pd.DataFrame):
    pt = pp.PercentChangeTransformer('f2', 'f1')
    result = pt.fit_transform(data)

    expected = np.array([0.0, 0.1, -0.02, 14.0, -0.7])

    nt.assert_array_equal(result, expected)
