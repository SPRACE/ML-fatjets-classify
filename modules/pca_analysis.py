from sklearn.decomposition import PCA


def pca_analysis(X, n):
    return PCA(n_components=n,
               svd_solver='randomized',
               random_state=1).fit_transform(X)


def plot_components(X, y):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context('talk')
    import pandas as pd
    df = pd.DataFrame(X)
    df['y'] = y
    sns.pairplot(df, vars=[0, 1, 2], hue='y', markers='.', diag_kind='kde')
    plt.suptitle('Pairplot of the first three principal components')
    return plt


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit("\n Wrong arguments \n")
    else:
        datadir, ptbin = sys.argv[1], sys.argv[2]

    from get_data import get_image, tidy_data
    sig, bkg = get_image(datadir, ptbin)
    X, y = tidy_data(sig, bkg)
    X_pca = pca_analysis(X, 60)

    print("Original matrix: ", X.shape)
    print("Transformed:     ", X_pca.shape)

    plot_components(X_pca, y).savefig("pca_components.png")
