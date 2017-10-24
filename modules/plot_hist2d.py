"""
    Nice plots for hyper-parameter tunning
    Requires python 2.7 and ROOT 6.10/08
"""

from __future__ import division

from ROOT import gROOT, TCanvas, TH1F, TH2F

gROOT.SetBatch(True)

timing = [25.431588, 26.897939, 26.796903, 26.496365, 25.870068, 28.00498, 27.415048, 27.883523, 28.471792, 11.837657, 13.906908, 13.91895, 13.11469, 12.009645, 14.420091, 14.410136, 10.762475, 22.237426, 7.745069, 7.478949, 5.469388, 5.340828, 5.563038, 8.192999, 8.092446, 7.739891, 9.885831, 3.689973, 4.315379, 4.488952, 4.49597, 4.411371, 4.306656, 4.76333, 4.652557, 6.712543, 2.784651, 2.691034, 2.481928, 2.031466, 2.100144, 2.92487, 3.040624, 2.696857, 3.816461, 1.371298, 1.932289, 1.98239, 2.016415, 2.018611, 1.932864, 2.134436, 2.224911, 2.929641, 1.101269, 1.194995, 1.104498, 1.098024, 1.824869, 1.473829, 1.441535, 1.224806, 2.134142, 0.99364, 1.353517, 1.173817, 1.48317, 1.522952, 1.607448, 1.484371, 1.595986, 1.770029, 1.095776, 1.22235, 1.258162, 1.020105, 1.191887, 1.093623, 1.16393, 1.098436, 1.228443, 1.037958, 1.032545, 1.108806, 1.030508, 1.169585, 1.310701, 1.109929, 1.124568, 1.589484, 1.136748, 0.922036, 0.959491, 1.113578, 1.054682, 1.080956, 1.043033, 1.095827, 1.100615]

test_accuracy = [0.50962943, 0.71603137, 0.71364057, 0.71563292, 0.6800372, 0.71656263, 0.71297652, 0.71271086, 0.52357548, 0.50962943, 0.71895337, 0.71855491, 0.71589851, 0.71722674, 0.73210251, 0.71935183, 0.50962943, 0.50962943, 0.71032012, 0.71324211, 0.71138263, 0.71191394, 0.71921903, 0.71868771, 0.71696109, 0.73050869, 0.71948469, 0.72001594, 0.71988314, 0.72253948, 0.71443748, 0.71085137, 0.70487446, 0.72426617, 0.70872629, 0.70606989, 0.73050869, 0.73064154, 0.73050869, 0.73714972, 0.73037589, 0.72971177, 0.7283836, 0.73316509, 0.72014874, 0.72771949, 0.7283836, 0.73502457, 0.73462611, 0.73050869, 0.73183692, 0.73263383, 0.72014874, 0.73329794, 0.7313056, 0.72984463, 0.7313056, 0.73250097, 0.72240669, 0.72692257, 0.73024309, 0.73728251, 0.73475891, 0.73329794, 0.73449332, 0.73303229, 0.73449332, 0.73555583, 0.73449332, 0.73196971, 0.73635274, 0.73343074, 0.72718823, 0.73157126, 0.7285164, 0.73011023, 0.73050869, 0.72811794, 0.72944611, 0.73157126, 0.72798514, 0.7071324, 0.70939034, 0.72612566, 0.7285164, 0.72811794, 0.72586, 0.7311728, 0.73011023, 0.70753086, 0.70606989, 0.71550006, 0.71430469, 0.68960023, 0.7121796, 0.71151549, 0.71430469, 0.72333646, 0.6748572]

xl = ['[1]', '[2]', '[5]', '[10]', '[5, 5]', '[10, 5]', '[5, 10]', '[10, 10]', '[200, 200]']
yl = ['1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024']

nx = len(xl)
ny = len(yl)

c1 = TCanvas('c1', 'c1', 1400, 800)

h1 = TH2F('h1', 'Accuracy Score | #color[2]{Training Time (seconds)}', nx, 0, nx, ny, 0, ny)
h1.SetStats(0)
h1.SetMarkerSize(1.5)
h1.GetXaxis().SetTitle('number of hidden units per layer')
h1.GetYaxis().SetTitle('batch size')
h1.GetXaxis().CenterTitle()
h1.GetYaxis().CenterTitle()
h1.GetXaxis().SetLabelSize(.050)
h1.GetYaxis().SetLabelSize(.050)
h1.GetXaxis().SetTitleSize(.045)
h1.GetYaxis().SetTitleSize(.045)

h2 = h1.Clone('h2')

h3 = TH1F('h3', 'Training Time', ny, 0, ny)
h3.SetStats(0)
h3.SetMarkerStyle(22)
h3.GetXaxis().SetTitle('batch size')
h3.GetYaxis().SetTitle('training time (seconds)')
h3.GetXaxis().CenterTitle()
h3.GetYaxis().CenterTitle()
h3.GetXaxis().SetLabelSize(.050)
h3.GetYaxis().SetLabelSize(.050)
h3.GetXaxis().SetTitleSize(.045)
h3.GetYaxis().SetTitleSize(.045)


for j in xrange(ny):
    h3.Fill(yl[j], timing[j*nx+2])
    for i in xrange(nx):
        z = test_accuracy[j*nx+i]
        t = timing[j*nx+i]
        h1.Fill(xl[i], yl[j], round(z, 2))
        h2.Fill(xl[i], yl[j], round(t, 0))

h1.Draw('colz')
h1.SetBarOffset(0.2)
h1.Draw('text same')
h2.SetBarOffset(-0.2)
h2.SetMarkerColor(2)
h2.Draw('text same')
c1.SaveAs('testaccuracy.png')

h3.Draw("hist")
c1.SaveAs('projection.png')
