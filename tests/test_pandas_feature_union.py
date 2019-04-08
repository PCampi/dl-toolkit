"""Test the pandas wrapper class."""

import numpy as np
import numpy.testing as nt
import pandas as pd
import pandas.testing as pt
import pytest
from scipy import sparse
import sklearn.pipeline as pline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

import src.preprocessing as pp


@pytest.fixture
def data():
    data = {
        'f1': np.array([100, 110, 98, 1500, 30]),
        'f2': 100 * np.ones((5, )),
        'f3': np.zeros((5, )),
        'target1': 100 + np.arange(5),
        'target2': 200 + np.arange(5),
        'income': [0, 0, 500000, 400000, 0],
    }

    return pd.DataFrame(data)


def test_it_concats_data(data: pd.DataFrame):
    transformer = pp.PandasFeatureUnion(
        [('1',
          pline.Pipeline(steps=[
              ('col_select',
               pp.ColumnSelector(['f1', 'f2', 'f3', 'target1', 'target2'])),
          ])),
         ('2',
          pline.Pipeline(steps=[('col_select',
                                 pp.ColumnSelector('income'))]))])

    result = transformer.fit_transform(data)
    expected = data

    pt.assert_frame_equal(expected, result)

    transformer = pp.PandasFeatureUnion(
        [('1',
          pline.Pipeline(steps=[
              ('col_select',
               pp.ColumnSelector(['f1', 'f2', 'f3', 'target1', 'target2'])),
          ])),
         ('2',
          pline.Pipeline(steps=[('col_select',
                                 pp.ColumnSelector('income'))]))])

    result = transformer.fit(data).transform(data)
    expected = data

    pt.assert_frame_equal(expected, result)


def test_it_returns_zeros_if_no_transformers(data: pd.DataFrame):
    transformer = pp.PandasFeatureUnion([('1', None), ('2', None)])

    result = transformer.fit_transform(data)
    expected = np.empty((data.shape[0], 0))

    nt.assert_array_equal(expected, result)

    result = transformer.fit(data).transform(data)
    expected = np.empty((data.shape[0], 0))

    nt.assert_array_equal(expected, result)


def test_it_raises_on_sparse(data: pd.DataFrame):
    f2f3 = data.loc[:, ['f2', 'f3']]
    t1 = FunctionTransformer(lambda x: sparse.csr_matrix(f2f3), validate=False)
    transformer = pp.PandasFeatureUnion([('1', t1),
                                         ('2', pp.Log10Transformer(['f2']))])

    with pytest.raises(pp.SparseNotAllowedError):
        transformer.fit_transform(f2f3)

    transformer = pp.PandasFeatureUnion([('1', t1),
                                         ('2', pp.Log10Transformer(['f2']))])

    with pytest.raises(pp.SparseNotAllowedError):
        transformer.fit(f2f3).transform(f2f3)


def test_it_raises_on_not_dataframe(data):
    f2f3 = data.loc[:, ['f2', 'f3']]
    transformer = pp.PandasFeatureUnion([('1', MinMaxScaler())])

    with pytest.raises(TypeError):
        transformer.fit_transform(f2f3)

    with pytest.raises(TypeError):
        transformer.fit(f2f3).transform(f2f3)
