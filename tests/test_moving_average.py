"""Test the percentage column difference transformer."""

import numpy as np
import numpy.testing as nt
import pandas as pd
import pandas.testing as pt
import pytest

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


def test_it_checks_init_params():
    with pytest.raises(TypeError):
        pp.MovingAverageTransformer(0, 10)

    with pytest.raises(ValueError):
        pp.MovingAverageTransformer('', 10)

    with pytest.raises(TypeError):
        pp.MovingAverageTransformer('f1', '10')

    with pytest.raises(ValueError):
        pp.MovingAverageTransformer('f1', 1)

    with pytest.raises(ValueError):
        pp.MovingAverageTransformer('f1', 10, kind='expo')


def test_it_checks_window_len(data):
    l = data.shape[0]

    with pytest.raises(ValueError):
        mat = pp.MovingAverageTransformer('f1', window=l)
        mat.fit(data)


def test_it_computes_simple_moving_average(data):
    simple_ma = pp.MovingAverageTransformer('target1', window=2, kind='simple')

    result = simple_ma.fit_transform(data)
    desired = pd.DataFrame(
        data={'target1': np.array([np.nan, 100.5, 101.5, 102.5, 103.5])})

    pt.assert_frame_equal(result, desired)

    simple_ma = pp.MovingAverageTransformer('target1', window=3, kind='simple')

    result = simple_ma.fit_transform(data)
    desired = pd.DataFrame(
        data={'target1': np.array([np.nan, np.nan, 101, 102, 103])})

    pt.assert_frame_equal(result, desired)


def test_it_computes_exp_moving_average(data):
    exp_ma = pp.MovingAverageTransformer(
        'target1', window=3, kind='exponential')

    result = exp_ma.fit_transform(data)
    desired = data.loc[:, ['target1']].ewm(span=3, adjust=True).mean()

    pt.assert_frame_equal(result, desired)
