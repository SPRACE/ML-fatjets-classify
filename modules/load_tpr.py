from sklearn.metrics import auc
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

# Location of the binary files
DATADIR = "bin"

# PT of the data sample
PTBIN = "250-300"


def load_data():
    data = os.path.join(DATADIR, "TruePositiveRates_" + PTBIN + ".npy")
    return np.load(data)


def plot(data):
    import matplotlib.pyplot as plt
    fpr = np.linspace(0, 1, 100)
    for tpr in data:
        area = auc(fpr, tpr)
        plt.plot(fpr, tpr, label="auc = {:.2f}".format(area))
    plt.legend(loc="lower right", title="area under curve (auc)")
    plt.title("Convolutional Neural Network")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    return plt


if __name__ == '__main__':
    data = load_data()
    plot(data).savefig("roc_cnn.png")
