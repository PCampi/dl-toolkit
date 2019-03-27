"""Test the base transformer via one of its subclasses selector."""

import numpy as np
import numpy.testing as nt
import pandas as pd
import pandas.testing as pt
import pytest

import src.preprocessing as pp


@pytest.fixture
def data() -> pd.DataFrame:
    data = {
        'f1': np.arange(5),
        'f2': np.arange(10, 15),
        'target1': 100 + np.arange(5),
        'target2': 200 + np.arange(5),
    }

    return pd.DataFrame(data)


def test_it_raises_wrong_str_init():
    bp = pp.BasePandasTransformer()

    with pytest.raises(TypeError):
        bp._check_str(1, 'NAME')

    with pytest.raises(ValueError):
        bp._check_str('', 'NAME')


def test_it_raises_wrong_X_type(data):
    bp = pp.BasePandasTransformer()

    wrong_X = data.values
    with pytest.raises(TypeError):
        bp.check_types(wrong_X)


def test_it_raises_wrong_y_type(data):
    bp = pp.BasePandasTransformer()

    wrong_y = data['f1'].values.tolist()
    with pytest.raises(TypeError):
        bp.check_types(data, wrong_y, y_type=np.ndarray)


def test_it_raises_not_implemented(data):
    bp = pp.BasePandasTransformer()

    with pytest.raises(NotImplementedError):
        bp._check_init_params(data)
