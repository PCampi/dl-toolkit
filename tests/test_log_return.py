"""Test the log return transformer."""

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


def test_it_raises_wrong_init():
    with pytest.raises(ValueError):
        pp.LogReturnsTransformer('')

    with pytest.raises(TypeError):
        pp.LogReturnsTransformer(1)

    with pytest.raises(TypeError):
        pp.LogReturnsTransformer(False)

    with pytest.raises(TypeError):
        pp.LogReturnsTransformer(0.52)


def test_it_computes_returns(data):
    ground_log: pd.DataFrame = np.log(data.loc[:, ['f2']])
    ground = ground_log.diff(periods=1)
    ground.columns = ['f2_log_ret']

    lr_transformer = pp.LogReturnsTransformer('f2')
    result = lr_transformer.fit_transform(data)

    pt.assert_frame_equal(ground, result)
