"""Test the log transformer."""

import numpy as np
import numpy.testing as nt
import pandas as pd
import pandas.testing as pt
import pytest

import src.preprocessing as pp


@pytest.fixture
def data():
    data = {
        'f1': np.arange(5),
        'f2': np.arange(10, 15),
        'target1': 100 + np.arange(5),
        'target2': 200 + np.arange(5),
    }

    return pd.DataFrame(data)


def test_it_raises_wrong_init_params(data):
    with pytest.raises(TypeError):
        pp.LogTransformer(1)

    with pytest.raises(ValueError):
        pp.LogTransformer('')


def test_it_raises_log_zero(data):
    lt = pp.LogTransformer('f1')

    with pytest.raises(ValueError):
        lt.fit_transform(data)


def test_it_makes_log(data):
    lt = pp.LogTransformer('f2')

    result = lt.fit(data).transform(data)
    expected: pd.DataFrame = np.log(data.loc[:, ['f2']])

    nt.assert_array_equal(result.values, expected.values)
    pt.assert_frame_equal(result, expected)


def test_it_makes_inverse_transform(data):
    lt = pp.LogTransformer('f2')

    log = lt.fit_transform(data)
    result = lt.inverse_transform(log)
    expected = data.loc[:, ['f2']].astype(np.float)

    nt.assert_almost_equal(result.values, expected.values)
    pt.assert_frame_equal(result, expected)
