from pandas import read_csv
import numpy as np
import os

# matplotlib without interactive UI
import matplotlib
matplotlib.use('Agg')

try:
    # OPTIONAL
    import seaborn
    seaborn.set_context('talk')
except ImportError:
    pass

# Location of the data directory
DATADIR = "data"

# PT of the data sample
PTBIN = "250-300"

# Sample names
SIG = os.path.join(DATADIR, "signal_PU0_13TeV_MJ-65-95_PTJ-" + PTBIN + ".txt")
BKG = os.path.join(DATADIR, "backgr_PU0_13TeV_MJ-65-95_PTJ-" + PTBIN + ".txt")

# Models names
MODELS = ["Multilayer Perceptron", "Logistic Regression", "Random Forest"]


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
              random_state=42)
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


def training(clf, X_train, X_test, y_train, y_test):
    # Import metrics
    from sklearn.metrics import roc_curve, auc
    # Fit model
    clf.fit(X_train, y_train)
    # Prepare result
    score_train = clf.score(X_train, y_train)
    score_test = clf.score(X_test, y_test)
    y_prob = clf.predict_proba(X_test)
    fpr, tpr, __ = roc_curve(y_test, y_prob[:, 1])
    fpr, tpr = interpolate_rates(fpr, tpr)
    return [fpr, tpr, auc(fpr, tpr), score_train, score_test]


def classifiers(X_train, X_test, y_train, y_test):
    # Import some classifiers
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    # Dictionary to save the results
    results = {}

    # Multilayer perceptron
    clf = MLPClassifier(solver='lbfgs',
                        hidden_layer_sizes=(5, ),
                        random_state=42)
    result = training(clf, X_train, X_test, y_train, y_test)
    results[MODELS[0]] = [0] + result

    # Logistic Regression
    clf = LogisticRegression(random_state=42)
    result = training(clf, X_train, X_test, y_train, y_test)
    results[MODELS[1]] = [1] + result

    # Random Forest
    clf = RandomForestClassifier(n_estimators=24,
                                 max_depth=9,
                                 random_state=42)
    result = training(clf, X_train, X_test, y_train, y_test)
    results[MODELS[2]] = [2] + result

    return results


def load_cnn_scores():
    data = os.path.join("bin", "ScoreTrain_" + PTBIN + ".npy")
    score_train = np.load(data)
    data = os.path.join("bin", "ScoreTest_" + PTBIN + ".npy")
    score_test = np.load(data)
    score_train_mean = np.mean(score_train)
    score_train_error = np.std(score_train)
    score_test_mean = np.mean(score_test)
    score_test_error = np.std(score_test)
    return [score_train_mean, score_train_error,
            score_test_mean, score_test_error]


def plot_metrics(X, y):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(16, 6))
    fig.suptitle("Model evaluation via stratified shuffle split")
    # Dictionary to put the scores
    score_train = {model: [] for model in MODELS}
    score_test = {model: [] for model in MODELS}
    # Iterate over the splits
    for X_train, X_test, y_train, y_test in split_generator(X, y):
        # Iterate over the classifiers
        results = classifiers(X_train, X_test, y_train, y_test).items()
        for model, result in results:
            i, fpr, tpr, area = result[0], result[1], result[2], result[3]
            ax[i].plot(fpr, tpr, label="auc = {:.2f}".format(area))
            ax[i].set_title(model)
            score_train[model] += [result[4]]
            score_test[model] += [result[5]]
    # Adjust axis of roc curves
    for axi in ax:
        axi.set_xlabel("False Positive Rate")
        axi.set_ylabel("True Positive Rate")
        axi.legend(loc="lower right", title="area under curve (auc)")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig('model-evaluation.png')
    plt.close()
    # Mean score values and errors
    score_train_mean, score_test_mean = [0, 0, 0], [0, 0, 0]
    score_train_error, score_test_error = [0, 0, 0], [0, 0, 0]
    for i, m in enumerate(MODELS):
        score_train_mean[i] = np.mean(score_train[m])
        score_train_error[i] = np.std(score_train[m])
        score_test_mean[i] = np.mean(score_test[m])
        score_test_error[i] = np.std(score_test[m])
    # Append CNN scores
    cnn_scores = load_cnn_scores()
    score_train_mean = [cnn_scores[0]] + score_train_mean
    score_train_error = [cnn_scores[1]] + score_train_error
    score_test_mean = [cnn_scores[2]] + score_test_mean
    score_test_error = [cnn_scores[3]] + score_test_error
    # Accuracy plot
    __, ax = plt.subplots(1, 1, figsize=(12, 8))
    bars = np.array([3, 2, 1, 0])
    width = 0.35
    hist1 = ax.barh(bars, score_train_mean, width, xerr=score_train_error)
    hist2 = ax.barh(bars+width, score_test_mean, width, xerr=score_test_error)
    # Adjust axis
    ax.set_title('Model Evaluation')
    ax.set_xlabel('Score')
    ax.set_xlim([.60, .85])
    ax.set_yticks(bars + 0.5*width)
    ax.set_yticklabels(["Convolutional Neural Net"] + MODELS)
    # Add legend
    ax.legend((hist2, hist1), ('Test', 'Train'))
    plt.tight_layout()
    plt.savefig('accuracy-score.png')


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
    plot_metrics(X_pca, y)
    print("\nResults:\nmodel-evaluation.png\naccuracy-score.png\n")
