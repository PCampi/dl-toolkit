from typing import List, Tuple

import numpy as np
import pandas as pd


def create_dataset(data: pd.DataFrame, features: List[str], bptt: int,
                   targets: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Create the dataset from the input time series.
    Data should already be normalized and preprocessed.

    Parameters
    ----------
    data: pd.DataFrame
        the data, where each row is a sample and each column a feature
    
    features: List[str]
        features to use
    
    targets: List[str]
        the target column(s)
    
    Returns
    ------
    np.ndarray:
        the X 3D tensor to use as input to a Keras RNN layer
    
    np.ndarray:
        the Y tensor to use as target for a Keras model
    """
    __check_input(data, features, targets)

    target_data: pd.DataFrame = data.loc[:, targets]
    rows, _ = data.shape

    if bptt >= rows:
        raise ValueError(
            f"BPTT ({bptt}) is greater than the available data points ({rows})"
        )

    ds = []
    y = []

    N = data.shape[0]
    start = 0
    end = start + bptt

    sliced = data.loc[:, features]

    while end < N:
        temp = sliced.iloc[start:end, :]
        ds.append(temp.values)
        y.append(target_data.iloc[end, :].values)
        start += 1
        end += 1

    X = np.array(ds)
    Y = np.array(y)

    try:
        ny = Y.shape[1]
        if ny == 1:
            Y = Y.flatten()
    except IndexError:
        Y = Y.flatten()

    return X, Y


def __check_input(data, features, targets):
    """Check if all arguments have the appropriate type and the
    requested target and features are in the data columns."""
    if not isinstance(features, list) or not all(
            isinstance(f, str) for f in features):
        raise TypeError(
            f"Expected a list of str as features, fot {type(features)}")

    if not all(f in data.columns for f in features):
        missing = set(features) - set(list(data.columns))
        s = f"""Some features are missing from the dataframe columns: {[x for x in missing]}"""
        raise ValueError(s)

    if not isinstance(targets, list):
        raise TypeError(
            f"Target should be a list of strings, got {type(targets)}")

    if isinstance(targets,
                  list) and not all(isinstance(t, str) for t in targets):
        raise TypeError("Target should be a list of strings")
    else:
        if not all(t in data.columns for t in targets):
            missing = set(targets) - set(list(data.columns))
            s = f"""Some targets are missing from the dataframe columns: {[x for x in missing]}"""
            raise ValueError(s)
