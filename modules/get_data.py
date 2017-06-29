import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")


def get_image(datadir, ptbin):
    sig = datadir + '/signal_PU0_13TeV_MJ-65-95_PTJ-' + ptbin + '.txt'
    bkg = datadir + '/backgr_PU0_13TeV_MJ-65-95_PTJ-' + ptbin + '.txt'
    return sig, bkg


def get_tau21(datadir, ptbin):
    sig = datadir + '/signal_PU0_13TeV_MJ-65-95_PTJ-' + ptbin + '_tau21.txt'
    bkg = datadir + '/backgr_PU0_13TeV_MJ-65-95_PTJ-' + ptbin + '_tau21.txt'
    return sig, bkg


def tidy_data(sig, bkg):
    X = pd.read_csv(sig, delim_whitespace=True)
    B = pd.read_csv(bkg, delim_whitespace=True)
    y = np.concatenate((np.ones(X.shape[0], dtype='bool'),
                        np.zeros(B.shape[0], dtype='bool')))
    X = np.concatenate((X, B))
    return X, y


def plot_tau21(X, y, ptbin):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context('talk')

    df = pd.DataFrame({'tau21': X.ravel(), 'y': y})

    sns.distplot(df.query('y== True').tau21, kde=True, color='g',
                 kde_kws={'label': 'W/Z process'})
    sns.distplot(df.query('y==False').tau21, kde=True, color='r',
                 kde_kws={'label': 'QCD process'})

    pt_str = '\t $P_{T}^{jet} = %s $ GeV,' % ptbin
    plt.title('$M_{jet} = 65-95$ GeV,' + pt_str + '\t Zero pileup')
    plt.xlabel('N-subjettiness')
    plt.ylabel('Normalized')
    return plt


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit("\n Wrong arguments \n")
    else:
        datadir, ptbin = sys.argv[1], sys.argv[2]

    sig, bkg = get_tau21(datadir, ptbin)
    X, y = tidy_data(sig, bkg)
    plot_tau21(X, y, ptbin).savefig("nsubjettiness.png")