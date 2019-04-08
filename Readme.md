# Dl toolset for financial time series analysis
This Python package aims to provide the basic building blocks for processing financial timeseries data using [Pandas](https://pandas.pydata.org) and [Scikit-learn](https://scikit-learn.org/stable/).

The rationale behind it is that Scikit-learn was not meant to work with Pandas `DataFrames`, so when you pass one to a sklearn transformer/pipeline, you loose all the Pandas goodies.
This package helps to retain all the convenience of Pandas while being consistent with the Scikit-learn's [Estimator API](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html).

This functionality is implemented in the sub-package named `preprocessing`.

The package also provides an utility sub-package, `rnn_input_creator`, which creates a dataset in the required format for Keras [Recurrent Neural Network](https://keras.io/layers/recurrent/) models, which require a 3D tensor with shape `(batch_size, timesteps, input_dim)`.

# Preprocessing
This is the main package.

It is composed of several preprocessing `Transformers`, which respect the Estimator API.
They all provide `fit`, `transform` and `fit_transform` methods.

The transformers work in a functional style: they **never** make in-place mutations of the DataFrames. Rather, they return a new DataFrame that you can use in subsequent transformations.
A convenience transformer, `DataFrameWrapper`, is provided to wrap operations that result in a numpy array into a DataFrame, since many sklearn transformers return a numpy array.

You can then choose to merge all back into the original DataFrame, or create a new one each time.
You are in control, because you know better.

The transformers are compatible with scikit-learn [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).


## Features

- opinionated design: all the supplied transformers work with pandas DataFrames as input, and output pandas DataFrames. This is so you can chain many of them in sequence inside a Pipeline and be sure that it will never complain about types. If you have a sklearn transformer and want to wrap the result in a DataFrame, use the `DataFrameWrapper` helper class. It just needs the final column names list.
- no in-place mutation of data: your initial dataframe is safe and sound, unless you decide to mutate it by yourself. As I said, you know better.

# rnn_input_creator
This package provides an input creator for feeding RNNs in Keras.

## Features

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
