"""Transformers which operate on a single column."""

from typing import List, Tuple, Type, TypeVar, Union, Callable, Sequence

import numpy as np
import pandas as pd

from .base import BasePandasTransformer


class LogReturnsTransformer(BasePandasTransformer):
    """Take the log return."""

    def __init__(self, columns: Sequence[str]):
        super().__init__(columns)

    def fit(self, X: pd.DataFrame, y=None) -> Type['LogReturnsTransformer']:
        super().prepare_to_fit(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        log: pd.DataFrame = np.log(X.loc[:, self.columns])
        result = log.diff(periods=1)
        result.columns = [f"{old_name}_log_ret" for old_name in result.columns]

        return result
