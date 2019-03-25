"""Test the RNN input creator module."""

import numpy as np
import numpy.testing as nt
import pandas as pd
import pytest

import rnn_input_creator as rn


@pytest.fixture
def data():
    data = {
        'f1': np.arange(5),
        'f2': np.arange(10, 15),
        'target1': 100 + np.arange(5),
        'target2': 200 + np.arange(5),
    }

    return pd.DataFrame(data)


def test_single_target(data):
    x_hat, y_hat = rn.create_dataset(data, ['f1', 'f2'], 3, ['target1'])

    x_expected = np.array([[[0, 10], [1, 11], [2, 12]],
                           [[1, 11], [2, 12], [3, 13]]])
    y_expected = np.array([103, 104])

    nt.assert_array_equal(y_hat, y_expected)
    nt.assert_array_equal(x_hat, x_expected)


def test_multiple_targets(data):
    x_hat, y_hat = rn.create_dataset(data, ['f1', 'f2'], 3,
                                     ['target1', 'target2'])

    x_expected = np.array([[[0, 10], [1, 11], [2, 12]],
                           [[1, 11], [2, 12], [3, 13]]])
    y_expected = np.array([[103, 203], [104, 204]])

    nt.assert_array_equal(y_hat, y_expected)
    nt.assert_array_equal(x_hat, x_expected)


def test_raises_on_wrong_features(data):
    with pytest.raises(ValueError):
        rn.create_dataset(data, ['f1', 'f2', 'f3'], 3, ['target1', 'target2'])

    with pytest.raises(ValueError):
        rn.create_dataset(data, ['f1', 'f3'], 3, ['target1', 'target2'])


def test_raises_on_str_target(data):
    with pytest.raises(TypeError):
        rn.create_dataset(data, ['f1', 'f2'], 3, 'target1')


def test_raises_on_wrong_target(data):
    with pytest.raises(ValueError):
        rn.create_dataset(data, ['f1', 'f2'], 3, ['target1', 'target3'])


def test_raises_on_bptt_too_large(data):
    with pytest.raises(ValueError):
        bptt = data.shape[0]
        rn.create_dataset(data, ['f1', 'f2'], bptt, ['target1'])

    with pytest.raises(ValueError):
        bptt = data.shape[0] + 1
        rn.create_dataset(data, ['f1', 'f2'], bptt, ['target1'])
