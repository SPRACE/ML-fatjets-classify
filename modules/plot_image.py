from get_data import get_image, tidy_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_image(data, ptbin, label):
    X = pd.read_csv(data, delim_whitespace=True)
    image = X.mean().values.reshape((25, 25))
    sns.heatmap(image, vmin=0, vmax=1)
    pt_str = '\t $P_{T}^{jet} = %s $ GeV,\t' % ptbin
    plt.title('Jet image: $25\\times25 $ cells,' + pt_str + label)
    plt.axis('off')
    return plt


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        sys.exit('\n Wrong arguments \n')
    else:
        datadir, ptbin = sys.argv[1], sys.argv[2]

    sns.set_context('talk')
    sig, bkg = get_image(datadir, ptbin)
    plot_image(sig, ptbin, '<signal>').savefig('signal_image.png')
    plt.close()
    plot_image(bkg, ptbin, '<background>').savefig('backgr_image.png')
