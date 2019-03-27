"""Test the pandas wrapper class."""

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


def test_it_checks_init_params():
    with pytest.raises(TypeError):
        pp.DataFrameWrapper(columns='age')

    with pytest.raises(TypeError):
        pp.DataFrameWrapper(columns='')

    with pytest.raises(ValueError):
        pp.DataFrameWrapper(columns=[])

    with pytest.raises(ValueError):
        pp.DataFrameWrapper(columns=['age', ''])


def test_it_raises_on_wrong_type(data):
    with pytest.raises(TypeError):
        wrapper = pp.DataFrameWrapper(['f1', 'f2'])
        wrapper.fit(data.loc[:, ['f1', 'f2']])


def test_it_raises_on_wrong_dimensions():
    X = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        wrapper = pp.DataFrameWrapper(['f1'])
        wrapper.fit(X)

    X = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError):
        wrapper = pp.DataFrameWrapper(['f1'])
        wrapper.fit(X)

    with pytest.raises(ValueError):
        wrapper = pp.DataFrameWrapper(['f1', 'f2', 'f3'])
        wrapper.fit(X)


def test_it_wraps_data(data: pd.DataFrame):
    columns = ['f1', 'target1']
    X = data.loc[:, columns].values

    wrapper = pp.DataFrameWrapper(columns=columns)
    result = wrapper.fit_transform(X)

    pt.assert_frame_equal(result, data.loc[:, columns])
