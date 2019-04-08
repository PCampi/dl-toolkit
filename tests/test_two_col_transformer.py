"""Test the two column transformer."""

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
    def op(a, b):  # pragma: no cover
        return a

    with pytest.raises(ValueError):
        pp.TwoColumnsTransformer(['f1'], op, 'f11')

    with pytest.raises(ValueError):
        pp.TwoColumnsTransformer(['f1', 'f2', 'f3'], op, 'f11')

    with pytest.raises(TypeError):
        pp.TwoColumnsTransformer([0, 'age'], op, 'f11')

    with pytest.raises(TypeError):
        pp.TwoColumnsTransformer(['age', 0], op, 'f11')

    with pytest.raises(ValueError):
        pp.TwoColumnsTransformer(['age', ''], op, 'f11')

    with pytest.raises(ValueError):
        pp.TwoColumnsTransformer(['', 'income'], op, 'f11')

    with pytest.raises(TypeError):
        pp.TwoColumnsTransformer(['', 'income'], op, 12)

    with pytest.raises(ValueError):
        pp.TwoColumnsTransformer(['', 'income'], op, '')

    with pytest.raises(TypeError):
        pp.TwoColumnsTransformer(['age', 'income'], None, 'f11')


def test_it_checks_columns_in_df(data: pd.DataFrame):
    def op(a, b):  # pragma: no cover
        return a

    with pytest.raises(ValueError):
        pt = pp.TwoColumnsTransformer(['f1', 'target3'], op, 'f11')
        pt.fit(data)

    with pytest.raises(ValueError):
        pt = pp.TwoColumnsTransformer(['target3', 'f1'], op, 'f11')
        pt.fit(data)


def test_it_runs_safety_checks(data):
    def safety_a(a):
        if not np.all(a != 0):
            raise ValueError("ValueError")

    def safety_b(b):
        if not np.all(b != 101):
            raise ValueError("ValueError")

    with pytest.raises(ValueError):
        pt = pp.TwoColumnsTransformer(
            ['f3', 'target2'], lambda a, b: (b - a) / a, 'percent_change',
            (safety_a, safety_b))
        pt.fit(data)

    with pytest.raises(ValueError):
        pt = pp.TwoColumnsTransformer(
            ['f2', 'target1'], lambda a, b: (b - a) / a, 'percent_change',
            (safety_a, safety_b))
        pt.fit(data)


def test_it_runs_with_no_safety_checks(data):
    perc = pp.TwoColumnsTransformer(['f2', 'f1'], lambda a, b: (b - a) / a,
                                    'percent_change')

    result = perc.fit_transform(data)
    expected = pd.DataFrame(
        data=np.array([0.0, 0.1, -0.02, 14.0, -0.7]),
        columns=['percent_change'])

    pt.assert_frame_equal(result, expected)


def test_it_transforms_data(data: pd.DataFrame):
    def safety_a(a):
        return np.all(a != 0)

    perc = pp.TwoColumnsTransformer(['f2', 'f1'], lambda a, b: (b - a) / a,
                                    'percent_change', (safety_a, None))
    result = perc.fit_transform(data)

    expected = pd.DataFrame(
        data=np.array([0.0, 0.1, -0.02, 14.0, -0.7]),
        columns=['percent_change'])

    pt.assert_frame_equal(result, expected)
