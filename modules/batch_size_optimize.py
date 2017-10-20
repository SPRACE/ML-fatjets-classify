"""
    Batch size optimization.
    Analyses the training time and model accuracy for different batch sizes
"""

import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ShuffleSplit

"""
    Data sets
"""
SIG = 'signal_PU0_13TeV_MJ-65-95_PTJ-250-300.txt'
BKG = 'backgr_PU0_13TeV_MJ-65-95_PTJ-250-300.txt'


def load_data():
    df_sig = pd.read_csv(SIG, delim_whitespace=True)
    df_bkg = pd.read_csv(BKG, delim_whitespace=True)

    y_data = np.concatenate((np.ones(df_sig.shape[0], dtype=np.int),
                             np.zeros(df_bkg.shape[0], dtype=np.int)))
    x_data = np.concatenate((df_sig, df_bkg))
    return x_data, y_data


def shuffle_split(x_data, y_data):
    """
        Shuffle split for cross validation
        Test sample: 30 %
    """
    split = ShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, test_index in split.split(x_data, y_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        return x_train, x_test, y_train, y_test


def grid_search():
    param_grid = {'batch_size': [5, 50, 500, 5000],
                  'hidden_units': [[5], [10], [5, 10], [10, 10]]}
    return list(ParameterGrid(param_grid))


def main(x_data, y_data):
    x_train, x_test, y_train, y_test = shuffle_split(x_data, y_data)
    """
        Iterate over parameters grid
    """
    for grid in grid_search():
        """
            Define input functions
        """
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": x_train},
                y=y_train,
                batch_size=grid['batch_size'],
                num_epochs=1,
                shuffle=False)
        """
            Build NN classifier
        """
        feature_columns = [tf.feature_column.numeric_column("x", shape=[625])]
        model = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                           hidden_units=grid['hidden_units'],
                                           n_classes=2)
        """
            Train model
        """
        t = datetime.datetime.now()
        model.train(input_fn=train_input_fn)
        print("\nTraining time = {}\n".format(datetime.datetime.now() - t))
        """
            Evaluate accuracy
        """
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": x_test},
                y=y_test,
                batch_size=grid['batch_size'],
                num_epochs=1,
                shuffle=False)
        accuracy = model.evaluate(input_fn=test_input_fn)["accuracy"]
        print("\nTest Accuracy: {0:f}\n".format(accuracy))


if __name__ == "__main__":
    x_data, y_data = load_data()
    main(x_data, y_data)
