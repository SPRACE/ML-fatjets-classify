import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')


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
