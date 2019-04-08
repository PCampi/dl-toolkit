"""Transformers used just for IO, they do not operate on data."""

import pandas as pd  # pragma: no cover

from .base import BasePandasTransformer  # pragma: no cover


class CsvSaver(BasePandasTransformer):  # pragma: no cover
    """Save the supplied DataFrame to a file."""

    def __init__(self, filename: str, write_index=False):
        self._check_init_params(filename)

        self.filename = filename
        self.write_index = write_index

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Save a dataframe to a file.

        Warning: it overwrites the file if it already exists.
        """
        X.to_csv(self.filename, index=self.write_index)
        return X

    def _check_init_params(self, filename: str):
        if not isinstance(filename, str):
            raise TypeError(f"filename must be a string, got {type(filename)}")

        if filename == '':
            raise ValueError("filename cannot be the empty string")
