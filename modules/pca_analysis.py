from sklearn.decomposition import PCA
from get_data import get_image, tidy_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pca_analysis(X, n):
    return PCA(n_components=n,
               whiten=True,
               svd_solver='randomized',
               random_state=42).fit_transform(X)


def plot_components(X, y):
    df = pd.DataFrame(X)
    df['Signal'] = y
    sns.pairplot(df, vars=[0, 1, 2],
                 hue='Signal', markers='.', diag_kind='kde')
    plt.suptitle('Pairplot of the first three principal components')
    return plt


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        sys.exit('\n Wrong arguments \n')
    else:
        datadir, ptbin = sys.argv[1], sys.argv[2]

    sns.set_context('talk')
    sig, bkg = get_image(datadir, ptbin)
    X, y = tidy_data(sig, bkg)
    X_pca = pca_analysis(X, 60)
    plot_components(X_pca, y).savefig('pca_components.png')
