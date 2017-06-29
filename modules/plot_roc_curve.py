import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit


def evaluate_model(X, y, model):
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=1)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        probas_ = model.fit(X_train, y_train).predict_proba(X_test)
        # Compute ROC curve
        fpr, tpr, __ = roc_curve(y_test, probas_[:, 1])
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

    mean_tpr /= sss.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    return mean_tpr, mean_fpr, mean_auc


def plot_roc_curve(X_img, y_img, X_tau, y_tau, ptbin):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context('talk')

    from pca_analysis import pca_analysis
    X_pca = pca_analysis(X_img, 60)

    from model_definition import model_definition
    names, models = model_definition()

    for name, model in zip(names, models):
        tpr, fpr, roc_auc = evaluate_model(X_pca, y_img, model)
        plt.plot(fpr, tpr, label=name + ' (area = %0.2f)' % roc_auc)

    tpr, fpr, roc_auc = evaluate_model(X_tau, y_tau, models[2])

    plt.plot(fpr, tpr, label='n-subjettiness (area = %0.2f)' % roc_auc)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    pt_str = '\t $P_{T}^{jet} = %s $ GeV,' % ptbin
    plt.title('$M_{jet} = 65-95$ GeV,' + pt_str + '\t Zero pileup')
    plt.legend(loc="lower right")
    return plt


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit("\n Wrong arguments \n")
    else:
        datadir, ptbin = sys.argv[1], sys.argv[2]

    from get_data import get_image, get_tau21, tidy_data
    sig_img, bkg_img = get_image(datadir, ptbin)
    sig_tau, bkg_tau = get_tau21(datadir, ptbin)
    X_img, y_img = tidy_data(sig_img, bkg_img)
    X_tau, y_tau = tidy_data(sig_tau, bkg_tau)
    plot_roc_curve(X_img, y_img, X_tau, y_tau, ptbin).savefig("roc_curve.png")
