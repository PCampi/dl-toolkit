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


def test_it_saves_columns_and_index_on_fit(data: pd.DataFrame):
    bp = pp.BasePandasTransformer()
    bp.prepare_to_fit(data)

    pt.assert_index_equal(bp.columns_, data.columns)
    pt.assert_index_equal(bp.index_, data.index)


def test_it_raises_empty_dataframe(data: pd.DataFrame):
    empty_df = data.drop(columns=data.columns)

    with pytest.raises(pp.EmptyDataFrameError):
        pp.BasePandasTransformer().prepare_to_fit(empty_df)


def test_it_wraps_single_column(data: pd.DataFrame):
    bp = pp.BasePandasTransformer()
    bp.prepare_to_fit(data[['f1']])

    assert bp.columns_ == ['f1']


def test_it_raises_wrong_X_type(data):
    with pytest.raises(TypeError):
        wrong_X = data.values  # a np.ndarray
        bp = pp.BasePandasTransformer()
        bp.prepare_to_fit(wrong_X)

    with pytest.raises(TypeError):
        wrong_X = data['f1']  # a pd.Series
        bp = pp.BasePandasTransformer()
        bp.prepare_to_fit(wrong_X)


def test_it_raises_wrong_y_type(data):
    bp = pp.BasePandasTransformer()

    with pytest.raises(TypeError):
        wrong_y = data['f1'].values.tolist()
        bp.prepare_to_fit(data, wrong_y)

    with pytest.raises(TypeError):
        wrong_y = data['f1']
        bp.prepare_to_fit(data, wrong_y)


def test_it_checks_str():
    bp = pp.BasePandasTransformer()

    with pytest.raises(TypeError):
        bp._check_str(1, 'name')

    with pytest.raises(ValueError):
        bp._check_str('', 'empty_str')
