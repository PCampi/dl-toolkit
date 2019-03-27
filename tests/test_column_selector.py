"""Test the column selector."""

import numpy as np
import numpy.testing as nt
import pandas as pd
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
    target_cols = ['f1', 10]
    with pytest.raises(AssertionError):
        pp.ColumnSelector(target_cols)

    target_cols = []
    with pytest.raises(AssertionError):
        pp.ColumnSelector(target_cols)

    target_cols = 'f1'
    with pytest.raises(AssertionError):
        pp.ColumnSelector(target_cols)

    target_cols = ['f2', '']
    with pytest.raises(AssertionError):
        pp.ColumnSelector(target_cols)

    target_cols = ['f1', 'f2', '']
    with pytest.raises(AssertionError):
        pp.ColumnSelector(target_cols)


def test_it_selects_columns(data):
    target_cols = ['f1', 'f2']
    t = pp.ColumnSelector(target_cols)

    result = t.fit(data).transform(data)
    result_cols = result.columns

    assert isinstance(result, pd.DataFrame)
    assert all(tc in result_cols for tc in target_cols)
    assert all(rc in target_cols for rc in result_cols)


def test_it_raises_missing_column(data):
    target_cols = ['f1', 'f3']
    t = pp.ColumnSelector(target_cols)

    with pytest.raises(AssertionError):
        t.fit(data)
