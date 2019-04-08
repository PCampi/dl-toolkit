"""Make a dataset for a study."""

from typing import List, Sequence, Type, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tqdm

from ..base import BasePandasTransformer


class DatasetCreator(BasePandasTransformer):
    def __init__(self,
                 columns: Sequence[str],
                 train_len: int,
                 bptt: int,
                 valid_percent: Union[float, None] = None,
                 shuffle_in_time=False,
                 shuffle_columns=False,
                 interactive=False):
        """Make a dataset inside a study period,
        starting from a pd.DataFrame ordered by index.
        
        The passed dataset must be the one considered for a *single* study period.

        Inside a study period, we take a training period of train_len, and
        inside the training period we move from the beginning onwards in blocks
        of window_len days.
        """
        self.__check_init_params(train_len, bptt, valid_percent,
                                 shuffle_in_time, shuffle_columns, interactive)

        super().__init__(columns)

        self.train_len = train_len
        self.bptt = bptt
        self.valid_percent = valid_percent
        self.shuffle_in_time = shuffle_in_time
        self.shuffle_columns = shuffle_columns
        self.interactive = interactive

    def fit(self, X: pd.DataFrame, y=None) -> Type['DatasetCreator']:
        """Check dimensions and prepare to fit.

        Most importantly, keep track of the names of the companies
        that were included in the S&P500 at the end of the training period,
        to keep only them in the training and testing phases.

        This is to follow Fischer, Krausse 2018.
        """
        self.prepare_to_fit(X)
        nr, _ = X.shape

        if nr <= self.train_len:
            raise ValueError("there are no days left for testing")

        if nr <= self.bptt:
            raise ValueError("dataset is smaller than bptt")

        last_train_day = X.iloc[self.train_len - 1, :].dropna()
        self.companies_included = last_train_day.index.tolist()

        return self

    def transform(self, X: pd.DataFrame):
        """Get the training and testing data, optionally also the validation data.

        IMPORTANT: the X variable should contain all the data for a
        **single** study period and not be longer, otherwise all indexes get messed up.

        In practice, X shall be a window of len = study_len inside the whole data,
        so that we can index it starting from 0 and get all indexes right.
        """
        final_data = X.loc[:, self.companies_included]
        n = final_data.shape[0]

        # TRAINING phase
        if self.valid_percent:
            X_train, X_valid, y_train, y_valid = self.subset_with_validation(
                data=final_data,
                start_index=self.bptt,
                end_index=self.train_len - 1,
                bptt=self.bptt,
                shuffle_columns=self.shuffle_columns,
                valid_percent=self.valid_percent)

            assert X_valid.shape[0] == y_valid.shape[0]
        else:
            X_train, y_train = self.subset_no_validation(
                data=final_data,
                start_index=self.bptt,
                end_index=self.train_len - 1,
                bptt=self.bptt,
                shuffle_columns=self.shuffle_columns)

        # TESTING phase
        X_test, y_test = self.subset_no_validation(
            data=final_data,
            start_index=self.train_len,
            end_index=n,
            bptt=self.bptt,
            shuffle_columns=False)

        # check dimensions
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

        # if shuffle_in_time, then shuffle *only* the training set
        if self.shuffle_in_time:
            n_train_samples, _, _ = X_train.shape
            permuted_indexes = np.random.permutation(n_train_samples)
            X_train = X_train[permuted_indexes]
            y_train = y_train[permuted_indexes]

        # return the computed data
        if self.valid_percent:
            return X_train, X_valid, X_test, y_train, y_valid, y_test
        else:
            return X_train, X_test, y_train, y_test

    def subset_no_validation(self, data: pd.DataFrame, start_index, end_index,
                             bptt, shuffle_columns):
        """Get a subset of the data with the rolling window method starting
        from "start_index", with a bptt of "bptt", and no validation data.
        """
        X_lst: List[np.ndarray] = []
        y_lst: List[np.ndarray] = []

        # for every slice i is the end index
        for_range = range(start_index, end_index)
        if self.interactive:
            for_range = tqdm.tqdm(for_range)

        for i in for_range:
            # slice_X has shape (n_companies, bptt)
            # slice_y has shape (n_companies)
            slice_X, slice_y = self._get_slice(
                data=data,
                start_index=i,
                bptt=bptt,
                shuffle_columns=shuffle_columns)

            # append to the resulting lists
            X_lst.append(slice_X)
            y_lst.append(slice_y)

        X: np.ndarray = np.concatenate(X_lst)  # shape=(n_train_samples, bptt)

        n_samples, n_columns = X.shape
        assert n_columns == bptt
        X = X.reshape(n_samples, n_columns, -1)

        y: np.ndarray = np.concatenate(y_lst)

        return X, y

    def subset_with_validation(self, data: pd.DataFrame, start_index,
                               end_index, bptt, shuffle_columns,
                               valid_percent: float):
        """Get a subset of the data with the rolling window method starting
        from "start_index", with a bptt of "bptt", and with validation data
        if valid_percent with respect to the data between start_index and
        end_index.
        """
        X_t: List[np.ndarray] = []
        X_v: List[np.ndarray] = []
        y_t: List[np.ndarray] = []
        y_v: List[np.ndarray] = []

        # for every slice i is the end index
        for_range = range(start_index, end_index)
        if self.interactive:
            for_range = tqdm.tqdm(for_range)

        for i in for_range:
            # slice_X has shape (n_companies, bptt)
            # slice_y has shape (n_companies)
            slice_X, slice_y = self._get_slice(
                data=data,
                start_index=i,
                bptt=bptt,
                shuffle_columns=shuffle_columns)

            # split data statifying on y
            tmp_X_train, tmp_X_valid, tmp_y_train, tmp_y_valid = train_test_split(
                slice_X,
                slice_y,
                test_size=self.valid_percent,
                stratify=slice_y)

            # append to the resulting lists
            X_t.append(tmp_X_train)
            X_v.append(tmp_X_valid)
            y_t.append(tmp_y_train)
            y_v.append(tmp_y_valid)

        X_train = np.concatenate(X_t)  # shape=(n_train_samples, bptt)
        X_valid = np.concatenate(X_v)  # shape=(n_valid_samples, bptt)

        n_train_samples, nc_t = X_train.shape
        assert nc_t == bptt
        X_train = X_train.reshape(n_train_samples, nc_t, -1)

        n_valid_samples, nc_v = X_valid.shape
        assert nc_v == bptt
        X_valid = X_valid.reshape(n_valid_samples, nc_v, -1)

        y_train = np.concatenate(y_t)
        y_valid = np.concatenate(y_v)

        return X_train, X_valid, y_train, y_valid

    def _get_slice(self, data: pd.DataFrame, start_index: int, bptt: int,
                   shuffle_columns: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Return a slice of the X and y data.

        X.shape[1] = y.shape[0]

        Returns
        -------
        X: np.ndarray
            array of shape (n_companies, bptt) where n_companies is the number
            of companies with valid sequence data in the whole input data
        
        y: np.ndarray
            array of shape (n_companies, ) with binary class labels (0, 1)
        """
        # the first bptt elements of the slice are the X, the last one
        # is the target y
        current_slice: pd.DataFrame = data.iloc[start_index -
                                                bptt:start_index + 1, :]

        # drop companies that don't have all points in this slice
        current_slice = current_slice.dropna(axis='columns', how='any')
        r, c = current_slice.shape

        if r == 0 or c == 0:
            raise ValueError(
                "there are no rows or no columns in the data after NaN deletion"
            )

        if shuffle_columns:
            col_indexes = np.random.permutation(c)
            X = current_slice.iloc[:-1, col_indexes].values
            returns = current_slice.iloc[-1, col_indexes].values
        else:
            X = current_slice.iloc[:-1, :].values
            returns = current_slice.iloc[-1, :].values
        # assign +1 and 0 classes if the value is above or below the slice median.
        # the median is the day median for the considered bptt step
        median = np.median(returns)

        y = (returns > median).astype(np.int)

        return X.transpose(), y

    def __check_init_params(self, train_len, bptt, valid_percent,
                            shuffle_in_time, shuffle_columns, interactive):
        """Check the input parameters."""
        if not isinstance(train_len, int):
            raise TypeError(f"train_len must be int, not {type(train_len)}")

        if bptt >= train_len:
            raise ValueError(
                f"bptt={bptt} and train_len={train_len}, should be the other way round"
            )

        if valid_percent < 0.0 or valid_percent > 0.99:
            raise ValueError(
                f"validation percent must be 0 <= v <= 0.99, got {valid_percent}"
            )

        if not isinstance(shuffle_in_time, bool):
            raise TypeError(
                f"shuffle_in_time must be bool, not {type(shuffle_in_time)}")

        if not isinstance(shuffle_columns, bool):
            raise TypeError(
                f"shuffle_columns must be bool, not {type(shuffle_columns)}")

        if not isinstance(interactive, bool):
            raise TypeError(
                f"interactive must be bool, not {type(interactive)}")
