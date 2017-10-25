"""
    Batch size optimization
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
    batch_size = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    hidden_units = [[1], [2], [5], [5, 5], [20, 20], [50, 50]]
    param_grid = {'batch_size': batch_size, 'hidden_units': hidden_units}
    return ParameterGrid(param_grid)


def main(x_data, y_data):
    x_train, x_test, y_train, y_test = shuffle_split(x_data, y_data)
    """
        Iterate over parameters grid
    """
    train_accuracy_results = []
    test_accuracy_results = []
    time_results = []
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
                                           n_classes=2,
                                           optimizer='Adam')
        """
            Train model
        """
        t = datetime.datetime.now()
        model.train(input_fn=train_input_fn)
        train_time = (datetime.datetime.now() - t).total_seconds()
        time_results += [train_time]
        """
            Evaluate accuracy
        """
        train_accuracy = model.evaluate(input_fn=train_input_fn)["accuracy"]
        train_accuracy_results += [train_accuracy]

        test_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": x_test},
                y=y_test,
                batch_size=grid['batch_size'],
                num_epochs=1,
                shuffle=False)

        test_accuracy = model.evaluate(input_fn=test_input_fn)["accuracy"]
        test_accuracy_results += [test_accuracy]

    print("\nTime results\n")
    print(time_results)

    print("\nTrain accuracy results\n")
    print(train_accuracy_results)

    print("\nTest accuracy results\n")
    print(test_accuracy_results)


if __name__ == "__main__":
    x_data, y_data = load_data()
    main(x_data, y_data)
