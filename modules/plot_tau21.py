from get_data import get_tau21, tidy_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_tau21(X, y, ptbin):
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


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        sys.exit('\n Wrong arguments \n')
    else:
        datadir, ptbin = sys.argv[1], sys.argv[2]

    sns.set_context('talk')
    sig, bkg = get_tau21(datadir, ptbin)
    X, y = tidy_data(sig, bkg)
    plot_tau21(X, y, ptbin).savefig('nsubjettiness.png')
