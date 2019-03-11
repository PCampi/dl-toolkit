# Input for RNNs in Keras

This package provides an input creator for feeding RNNs in Keras.

## Features - as user stories

The features are:
- splitting training, validation and testing
- shuffling
- creation of batches (dealing with non-uniform number of samples for the last batch)

## Interface

The main API should accept the input time-series as a pandas DataFrame, the window length for *BPTT* and the target column name.
The target column shall be contained in the dataframe and its name is specified in the function call.

For instance, the main function shold be similar to:

```python
def create_dataset(data: pd.DataFrame,
                   shuffle: bool,
                   target: str)
```
