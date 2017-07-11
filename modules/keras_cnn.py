from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_curve
import pandas as pd
import numpy as np
import datetime
import os

# Location of the data directory
DATADIR = "data"

# PT of the data sample
PTBIN = "250-300"

# Epochs
EPOCHS = 20

# Splits
SPLITS = 10

# Image size
IMAGESIZE = (25, 25, 1)

# Sample names
SIG = os.path.join(DATADIR, "signal_PU0_13TeV_MJ-65-95_PTJ-" + PTBIN + ".txt")
BKG = os.path.join(DATADIR, "backgr_PU0_13TeV_MJ-65-95_PTJ-" + PTBIN + ".txt")

# Output files
TPR_OUTFILE = "TruePositiveRates_" + PTBIN + ".npy"
SCORE_TRAIN = "ScoreTrain_" + PTBIN + ".npy"
SCORE_TEST = "ScoreTest_" + PTBIN + ".npy"

# Random seed
np.random.seed(42)


def load_data():
    df_sig = pd.read_csv(SIG, delim_whitespace=True)
    df_bkg = pd.read_csv(BKG, delim_whitespace=True)
    return df_sig, df_bkg


def concatenate_sig_bkg(df_sig, df_bkg):
    y = np.concatenate((np.ones(df_sig.shape[0], dtype='bool'),
                        np.zeros(df_bkg.shape[0], dtype='bool')))
    X = np.concatenate((df_sig, df_bkg))
    return X, y


def reshape_image(X):
    return X.reshape((-1, ) + IMAGESIZE)


def split_generator(X, y):
    # Stratified shuffle split for cross validation
    # Test sample = 30%
    split = StratifiedShuffleSplit(n_splits=SPLITS,
                                   test_size=0.3, random_state=42)
    for train_index, test_index in split.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yield X_train, X_test, y_train, y_test


def interpolate_rates(fpr, tpr):
    # ROC curve with 100 steps
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.interp(mean_fpr, fpr, tpr)
    return mean_tpr


def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict_classes(X_train, verbose=0)
    score_train = accuracy_score(y_train, y_pred)
    y_pred = model.predict_classes(X_test, verbose=0)
    score_test = accuracy_score(y_test, y_pred)
    y_prob = model.predict(X_test, verbose=0)[:, 1]
    fpr, tpr, __ = roc_curve(y_test, y_prob)
    tpr = interpolate_rates(fpr, tpr)
    return [tpr, score_train, score_test]


def simple_model():
    model = Sequential()
    model.add(Dense(units=256, kernel_initializer='uniform',
              activation='relu', input_dim=625))
    model.add(Dropout(0.2))
    model.add(Dense(units=256, kernel_initializer='uniform',
              activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=2, kernel_initializer='uniform',
              activation='softmax'))
    return 'simpleNN', model


def cnn_model():
    filters = 32
    kernel_size = 3
    pool_size = 2
    num_classes = 2
    feature_layers = [
        Conv2D(filters, kernel_size,
               input_shape=IMAGESIZE, activation="relu"),
        MaxPooling2D(pool_size),
        Dropout(0.25),
        Conv2D(64, kernel_size, activation="relu", padding='same'),
        Conv2D(64, kernel_size, activation="relu"),
        MaxPooling2D(pool_size),
        Dropout(0.25),
        Flatten(),
    ]
    classification_layers = [
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ]
    model = Sequential(feature_layers + classification_layers)
    return 'cNN', model


def run_model(X, y):
    # name, model = simple_model()
    name, model = cnn_model()

    model.compile(loss='binary_crossentropy',
                  optimizer='Adam')

    # Original weights
    model.save_weights('model.h5')

    # Lists to put the results
    tpr = []
    score_train = []
    score_test = []
    # Iterate over the splits
    for X_train, X_test, y_train, y_test in split_generator(X, y):
        if name == 'cNN':
            X_train = reshape_image(X_train)
            X_test = reshape_image(X_test)
        # Fit model
        model.fit(X_train, to_categorical(y_train, num_classes=2),
                  batch_size=50, epochs=EPOCHS, verbose=0)
        # Evaluate model
        results = evaluate_model(model, X_train, X_test, y_train, y_test)
        tpr += [results[0]]
        score_train += [results[1]]
        score_test += [results[2]]
        # Reset weights
        model.load_weights('model.h5')
    return tpr, score_train, score_test


if __name__ == '__main__':
    # Load the data sets
    print("\nLoading signal data from {}".format(SIG))
    print("\nLoading backgr data from {}".format(BKG))
    df_sig, df_bkg = load_data()

    # Process data
    print("\nProcessing {} signal and {} backgr samples".format(
           len(df_sig.index), len(df_bkg.index)))
    X, y = concatenate_sig_bkg(df_sig, df_bkg)

    # Create model
    print("\nTraining model")
    t = datetime.datetime.now()
    tpr, score_train, score_test = run_model(X, y)
    print("\nTraining time = {}".format(datetime.datetime.now() - t))
    score_train_mean = np.mean(score_train)
    score_train_error = np.std(score_train)
    score_test_mean = np.mean(score_test)
    score_test_error = np.std(score_test)
    print("\nTraining score = {:.4f} +/- {:.4f}".format(score_train_mean,
                                                        score_train_error))
    print("\nTesting score = {:.4f} +/- {:.4f}".format(score_test_mean,
                                                       score_test_error))
    print("\nWriting results to file {}".format(TPR_OUTFILE))
    # Save the true positive rates for all splits
    # The false positive rates are always np.linspace(0, 1, 100)
    np.save(TPR_OUTFILE, tpr)
    np.save(SCORE_TRAIN, score_train)
    np.save(SCORE_TEST, score_test)
