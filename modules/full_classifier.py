from pandas import read_csv
import numpy as np
import os

# matplotlib without interactive UI
import matplotlib
matplotlib.use('Agg')

try:
    # OPTIONAL
    import seaborn
except ImportError:
    pass

# Location of the data directory
DATADIR = "data"

# PT of the data sample
PTBIN = "250-300"

# Sample names
SIG = os.path.join(DATADIR, "signal_PU0_13TeV_MJ-65-95_PTJ-" + PTBIN + ".txt")
BKG = os.path.join(DATADIR, "backgr_PU0_13TeV_MJ-65-95_PTJ-" + PTBIN + ".txt")


def load_data():
    df_sig = read_csv(SIG, delim_whitespace=True)
    df_bkg = read_csv(BKG, delim_whitespace=True)
    return df_sig, df_bkg


def concatenate_sig_bkg(df_sig, df_bkg):
    y = np.concatenate((np.ones(df_sig.shape[0], dtype='bool'),
                        np.zeros(df_bkg.shape[0], dtype='bool')))
    X = np.concatenate((df_sig, df_bkg))
    return X, y


def reduce_dimensionality(X, n):
    # Principal component analysis
    # The option whiten=True ensures normalized components
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n,
              whiten=True,
              svd_solver='randomized',
              random_state=1)
    X_pca = pca.fit_transform(X)
    return X_pca


def split_generator(X, y):
    # Stratified shuffle split for cross validation
    # Number of splits = 10
    # Test sample = 30%
    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    for train_index, test_index in split.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yield X_train, X_test, y_train, y_test


def interpolate_rates(fpr, tpr):
    # ROC curve with 100 steps
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.interp(mean_fpr, fpr, tpr)
    return mean_fpr, mean_tpr


def classifiers(X_train, X_test, y_train, y_test):
    # Import some classifiers
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    # Import ROC curve metric
    from sklearn.metrics import roc_curve, auc

    # Dictionary to save the results
    results = {}

    # Multilayer perceptron
    clf = MLPClassifier(solver='lbfgs',
                        hidden_layer_sizes=(5, ),
                        random_state=42)
    y_prob = clf.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, __ = roc_curve(y_test, y_prob[:, 1])
    fpr, tpr = interpolate_rates(fpr, tpr)
    results["Multilayer Perceptron"] = [0, fpr, tpr, auc(fpr, tpr)]

    # Logistic Regression
    clf = LogisticRegression(random_state=42)
    y_prob = clf.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, __ = roc_curve(y_test, y_prob[:, 1])
    fpr, tpr = interpolate_rates(fpr, tpr)
    results["Logistic Regression"] = [1, fpr, tpr, auc(fpr, tpr)]

    # Random Forest
    clf = RandomForestClassifier(n_estimators=24,
                                 max_depth=9,
                                 random_state=42)
    y_prob = clf.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, __ = roc_curve(y_test, y_prob[:, 1])
    fpr, tpr = interpolate_rates(fpr, tpr)
    results["Random Forest"] = [2, fpr, tpr, auc(fpr, tpr)]

    return results


def custom_ax(ax, fpr, tpr, area, title):
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.plot(fpr, tpr, label="auc = {:.2f}".format(area))
    ax.legend(loc="lower right", title="area under curve (auc)")


def plot_roc_curves(X, y):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(16, 5))
    fig.suptitle("Model evaluation via stratified shuffle split")
    # Iterate over the splits
    for X_train, X_test, y_train, y_test in split_generator(X, y):
        # Iterate over the classifiers
        results = classifiers(X_train, X_test, y_train, y_test).items()
        for title, result in results:
            i, fpr, tpr, area = result[0], result[1], result[2], result[3]
            custom_ax(ax[i], fpr, tpr, area, title)

    return plt


if __name__ == '__main__':
    # Load the data sets
    print("\nLoading signal data from {}".format(SIG))
    print("\nLoading backgr data from {}".format(BKG))
    df_sig, df_bkg = load_data()

    # Process data
    print("\nProcessing {} signal and {} backgr samples".format(
           len(df_sig.index), len(df_bkg.index)))
    X, y = concatenate_sig_bkg(df_sig, df_bkg)

    # Reduce dimensionality
    X_pca = reduce_dimensionality(X, 60)
    print("\nDimensionality reduction from {} to {} attributes".format(
           X.shape[1], X_pca.shape[1]))

    # Evaluate classifiers
    print("\nEvaluating classifiers")
    plot_roc_curves(X_pca, y).savefig('ml-vs-nsubjettiness.png')
    print("\nResults: ml-vs-nsubjettiness.png\n")
