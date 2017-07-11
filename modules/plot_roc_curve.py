from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit
from pca_analysis import pca_analysis
from model_definition import model_definition
from get_data import get_image, get_tau21, tidy_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# PT of the data sample
PTBIN = "250-300"

# ROC Curve x-axis
MEAN_FPR = np.linspace(0, 1, 100)


def evaluate_model(X, y, model):
    split = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    mean_tpr = np.zeros(100)
    area = []
    for train_index, test_index in split.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        probas_ = model.fit(X_train, y_train).predict_proba(X_test)
        # Compute ROC curve
        fpr, tpr, __ = roc_curve(y_test, probas_[:, 1])
        tpr = np.interp(MEAN_FPR, fpr, tpr)
        mean_tpr += tpr
        area += [auc(MEAN_FPR, tpr)]

    mean_tpr /= split.get_n_splits(X, y)
    mean_auc = np.mean(area)
    error_auc = np.std(area)
    return mean_tpr, mean_auc, error_auc


def load_cnn():
    mean_tpr = np.zeros(100)
    data = os.path.join("bin", "TruePositiveRates_" + PTBIN + ".npy")
    tprs = np.load(data)
    area = []
    for tpr in tprs:
        mean_tpr += tpr
        area += [auc(MEAN_FPR, tpr)]

    mean_tpr /= 10  # 10 splits
    mean_auc = np.mean(area)
    error_auc = np.std(area)
    return mean_tpr, mean_auc, error_auc


def plot_roc_curve(X_img, y_img, X_tau, y_tau, ptbin):
    # Plot Convolutional NN first
    tpr, roc_auc, err = load_cnn()
    name = "convolutional neural net "
    plt.plot(MEAN_FPR, tpr,
             label=name + "(auc = {:.2f} +/- {:.3f})".format(roc_auc, err))
    # Plot other models
    X_pca = pca_analysis(X_img, 60)
    names, models = model_definition()
    for name, model in zip(names, models):
        tpr, roc_auc, err = evaluate_model(X_pca, y_img, model)
        plt.plot(MEAN_FPR, tpr,
                 label=name +
                 " (auc = {:.2f} +/- {:.3f})".format(roc_auc, err))
    # Plot n-subjettiness
    plt.plot([0.272], [0.710], '*', color='black', markersize=20,
             label='n-subjettiness ($\\tau_{21}<0.37$)')
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    pt_str = '\t $P_{T}^{jet} = %s $ GeV,' % ptbin
    plt.title('$M_{jet} = 65-95$ GeV,' + pt_str + '\t Zero pileup')
    plt.legend(loc='lower right')
    return plt


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        sys.exit("\n Wrong arguments \n")
    else:
        datadir, ptbin = sys.argv[1], sys.argv[2]

    sns.set_context('talk')
    sig_img, bkg_img = get_image(datadir, ptbin)
    sig_tau, bkg_tau = get_tau21(datadir, ptbin)
    X_img, y_img = tidy_data(sig_img, bkg_img)
    X_tau, y_tau = tidy_data(sig_tau, bkg_tau)
    plot_roc_curve(X_img, y_img, X_tau, y_tau, ptbin).savefig('roc_curve.png')
