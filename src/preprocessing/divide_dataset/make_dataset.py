"""Make a dataset for a study."""

from typing import List, Sequence, Type, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tqdm

from ..base import BasePandasTransformer


class DatasetCreator(BasePandasTransformer):
    """Create a dataset in the same way as Fischer 2018."""

    def __init__(self,
                 columns: Sequence[str],
                 train_len: int,
                 bptt: int,
                 interactive=False):
        """Make a dataset inside a study period,
        starting from a pd.DataFrame ordered by index.
        
        The passed dataset must be the one considered for a *single* study period.

        Inside a study period, we take a training period of train_len, and
        inside the training period we move from the beginning onwards in blocks
        of window_len days.
        """
        self.__check_init_params(train_len, bptt, interactive)
        super().__init__(columns)

        self.train_len = train_len
        self.bptt = bptt
        self.interactive = interactive

        self.has_transformed = False

        self.company_names_list_train: List[List[str]] = []
        self.forecast_time_list_train: List[np.ndarray] = []
        self.company_names_list_test: List[List[str]] = []
        self.forecast_time_list_test: List[np.ndarray] = []

        self.company_names_train: np.ndarray = None
        self.forecast_time_train: np.ndarray = None
        self.company_names_test: np.ndarray = None
        self.forecast_time_test: np.ndarray = None

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

        last_train_day: pd.Series = X.iloc[self.train_len - 1, :].dropna()
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
        X_train, y_train = self.get_subset(
            data=final_data,
            start_index=self.bptt,
            end_index=self.train_len,
            names=self.company_names_list_train,
            times=self.forecast_time_list_train)

        # TESTING phase
        X_test, y_test = self.get_subset(
            data=final_data,
            start_index=self.train_len,
            end_index=n,
            names=self.company_names_list_test,
            times=self.forecast_time_list_test)

        # check dimensions
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

        self.has_transformed = True

        self.forecast_time_train = np.concatenate(
            self.forecast_time_list_train).astype(np.int)
        self.company_names_train = np.concatenate(
            self.company_names_list_train)

        self.forecast_time_test = np.concatenate(
            self.forecast_time_list_test).astype(np.int)
        self.company_names_test = np.concatenate(self.company_names_list_test)

        assert self.forecast_time_train.shape[0] == X_train.shape[0]
        assert self.company_names_train.shape[0] == X_train.shape[0]

        assert self.forecast_time_test.shape[0] == X_test.shape[0]
        assert self.company_names_test.shape[0] == X_test.shape[0]

        # return the computed data
        return X_train, X_test, y_train, y_test

    def get_subset(self, data: pd.DataFrame, start_index, end_index,
                   names: List[np.ndarray], times: List[np.ndarray]):
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
                data=data, last_seq_index=i, names=names, times=times)

            # append to the resulting lists
            X_lst.append(slice_X)
            y_lst.append(slice_y)

        X: np.ndarray = np.concatenate(X_lst)  # shape=(n_train_samples, bptt)

        n_samples, n_columns = X.shape
        assert n_columns == self.bptt
        X = X.reshape(n_samples, n_columns, -1)

        y: np.ndarray = np.concatenate(y_lst)

        return X, y

    def _get_slice(
            self,
            data: pd.DataFrame,
            last_seq_index: int,
            names: List[np.ndarray],
            times: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        bptt = self.bptt
        current_slice: pd.DataFrame = data.iloc[last_seq_index -
                                                bptt:last_seq_index + 1, :]

        # drop companies that don't have all points in this slice
        current_slice = current_slice.dropna(axis='columns', how='any')
        r, c = current_slice.shape

        if r == 0 or c == 0:
            raise ValueError(
                "there are no rows or no columns in the data after NaN deletion"
            )

        times.append((last_seq_index + 1) * np.ones(c, ))

        X_tmp = current_slice.iloc[:-1, :]
        X = X_tmp.values

        returns = current_slice.iloc[-1, :].values

        current_slice_companies = X_tmp.columns.tolist()
        names.append(current_slice_companies)

        # assign +1 and 0 classes if the value is above or below the slice median.
        # the median is the day median for the considered bptt step
        median = np.median(returns)

        y = (returns > median).astype(np.int8)

        return X.transpose(), y

    def get_times_names(self):
        if not self.has_transformed:
            raise RuntimeError(
                "Transformer did not transform and there are no times and names"
            )

        result = {
            'train': {
                'time': self.forecast_time_train,
                'names': self.company_names_train,
            },
            'test': {
                'time': self.forecast_time_test,
                'names': self.company_names_test,
            }
        }

        return result

    def __check_init_params(self, train_len, bptt, interactive):
        """Check the input parameters."""
        if not isinstance(train_len, int):
            raise TypeError(f"train_len must be int, not {type(train_len)}")

        if bptt >= train_len:
            raise ValueError(
                f"bptt={bptt} and train_len={train_len}, should be the other way round"
            )

        if not isinstance(interactive, bool):
            raise TypeError(
                f"interactive must be bool, not {type(interactive)}")
