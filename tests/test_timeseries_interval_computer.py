"""Test the timeseries divider transformer."""

import numpy as np
import numpy.testing as nt
import pandas as pd
import pandas.testing as pt
import pytest

import src.preprocessing as pp
import src.preprocessing.divide_dataset as dd


def test_it_raises_wrong_date_col():
    l = 1000
    t = 750
    s = 250

    with pytest.raises(ValueError):
        dd.TimeseriesDateIntervalComputer('', l, s, t)

    with pytest.raises(TypeError):
        dd.TimeseriesDateIntervalComputer(1, l, s, t)

    with pytest.raises(TypeError):
        dd.TimeseriesDateIntervalComputer(['date'], l, s, t)

    with pytest.raises(TypeError):
        dd.TimeseriesDateIntervalComputer('date', 2.5, s, t)

    with pytest.raises(TypeError):
        dd.TimeseriesDateIntervalComputer('date', l, 2.5, t)

    with pytest.raises(ValueError):
        dd.TimeseriesDateIntervalComputer('date', -4, s, t)

    with pytest.raises(ValueError):
        dd.TimeseriesDateIntervalComputer('date', l, 0, t)

    with pytest.raises(TypeError):
        dd.TimeseriesDateIntervalComputer('date', l, s, '250')

    with pytest.raises(ValueError):
        dd.TimeseriesDateIntervalComputer('date', l, s, l)
