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


def test_it_raises_wrong_init_params():
    with pytest.raises(TypeError):
        pp.BasePandasTransformer(None)

    with pytest.raises(TypeError):
        pp.BasePandasTransformer(1)

    with pytest.raises(ValueError):
        pp.BasePandasTransformer('')

    with pytest.raises(ValueError):
        pp.BasePandasTransformer([''])

    with pytest.raises(ValueError):
        pp.BasePandasTransformer(['f1', ''])


def test_it_converts_init_columns(data: pd.DataFrame):
    bp = pp.BasePandasTransformer(('f1', 'f2'))
    assert bp.columns == ['f1', 'f2']

    bp = pp.BasePandasTransformer('f1')
    assert bp.columns == ['f1']

    bp = pp.BasePandasTransformer(('f1'))
    assert bp.columns == ['f1']

    bp = pp.BasePandasTransformer(data.columns)
    assert isinstance(bp.columns, list)
    assert bp.columns == data.columns.tolist()


def test_it_raises_wrong_X_type(data):
    bp = pp.BasePandasTransformer(('f1'))

    wrong_X = data.values
    with pytest.raises(TypeError):
        bp.prepare_to_fit(wrong_X)


def test_it_raises_wrong_y_type(data):
    bp = pp.BasePandasTransformer('f1')

    wrong_y = data['f1'].values.tolist()
    with pytest.raises(TypeError):
        bp.prepare_to_fit(data, wrong_y)


def test_it_prepares_to_fit(data):
    bp = pp.BasePandasTransformer('f1')
    bp.prepare_to_fit(data)

    pt.assert_index_equal(bp.index_, data.index)
