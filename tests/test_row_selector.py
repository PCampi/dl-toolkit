"""Test the column selector."""

import numpy as np
import numpy.testing as nt
import pandas as pd
import pandas.testing as pt
import pytest

import preprocessing as pp


@pytest.fixture
def data() -> pd.DataFrame:
    data = {
        'f1': np.arange(5),
        'f2': np.arange(10, 15),
        'target1': 100 + np.arange(5),
        'target2': 200 + np.arange(5),
    }

    return pd.DataFrame(data)


def test_it_raises_wrong_init_params(data):
    start = 1
    end = 1

    with pytest.raises(AssertionError):
        pp.RowSelector(start, end)

    start = 2
    end = 1
    with pytest.raises(AssertionError):
        pp.RowSelector(start, end)

    start = 0
    end = -5
    with pytest.raises(AssertionError):
        t = pp.RowSelector(start, end)
        t.fit(data)

    start = 0
    end = data.shape[0] + 1
    with pytest.raises(AssertionError):
        t = pp.RowSelector(start, end)
        t.fit(data)

    start = -1
    end = 3
    with pytest.raises(AssertionError):
        t = pp.RowSelector(start, end)
        t.fit(data)

    with pytest.raises(AssertionError):
        pp.RowSelector(start='ciao', end=-1)
        pp.RowSelector(start=1, end=True)


def test_it_selects_rows(data: pd.DataFrame):
    rs = pp.RowSelector(start=0, end=2)

    sel = rs.fit(data).transform(data)
    expected = data.iloc[0:2, :]

    assert sel.shape == (2, len(data.columns))
    pt.assert_frame_equal(sel, expected)


def test_it_raises_start_too_high(data: pd.DataFrame):
    rs = pp.RowSelector(start=data.shape[0], end=-1)

    with pytest.raises(AssertionError):
        rs.fit(data)


def test_it_selects_last_row(data: pd.DataFrame):
    rs = pp.RowSelector(start=data.shape[0] - 1, end=-1)
    sel = rs.fit(data).transform(data)
    pt.assert_frame_equal(sel, data.iloc[[-1], :])

    rs = pp.RowSelector(start=-1, end=-1)
    sel = rs.fit(data).transform(data)
    pt.assert_frame_equal(sel, data.iloc[[-1], :])
