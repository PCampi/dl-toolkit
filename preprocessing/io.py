"""Transformers used just for IO, they do not operate on data."""

from typing import List, Tuple, Type, TypeVar, Union, Callable

import pandas as pd

from .base import BasePandasTransformer


class CsvSaver(BasePandasTransformer):
    """Save the supplied DataFrame to a file."""

    def __init__(self, filename: str):
        self._check_init_params(filename)

        self.filename = filename

    def transform(self, X: pd.DataFrame):
        """Save a dataframe to a file.

        Warning: it overwrites the file if it already exists.
        """
        X.to_csv(self.filename)
        return X

    def _check_init_params(self, filename: str):
        if not isinstance(filename, str):
            raise TypeError(f"filename must be a string, got {type(filename)}")

        if filename == '':
            raise ValueError("filename cannot be the empty string")
