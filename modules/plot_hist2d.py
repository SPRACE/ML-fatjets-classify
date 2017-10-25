"""
    Nice plots for hyper-parameter tunning
    Requires python 2.7 and ROOT 6.10/08
"""

from __future__ import division

from ROOT import gROOT, TCanvas, TH1F, TH2F

gROOT.SetBatch(True)

timing = [24.410439, 24.910075, 25.664182, 26.962138, 27.590327, 29.349585, 13.150066, 13.434365, 13.265842, 13.950145, 13.947584, 14.834151, 4.931589, 5.332372, 6.403859, 7.547976, 6.513756, 7.010882, 3.374669, 4.471188, 4.31369, 4.542263, 4.602837, 4.788853, 1.891384, 1.940521, 2.134539, 2.522599, 2.868164, 2.718478, 1.701724, 1.528077, 1.758688, 2.227819, 1.94363, 2.358413, 1.134119, 1.103618, 1.539358, 1.755239, 1.777297, 1.883475, 1.098762, 1.200488, 1.345682, 1.199053, 1.210265, 1.404781, 1.244428, 1.123855, 1.288538, 1.161341, 1.164904, 1.307494, 0.964987, 1.01816, 1.015437, 1.152123, 1.114411, 1.185826, 1.147678, 0.969598, 0.937501, 1.062772, 0.964091, 1.040038]

test_accuracy = [0.50962943, 0.71603137, 0.71364057, 0.6800372, 0.71895337, 0.50962943, 0.50962943, 0.71895337, 0.71855491, 0.71722674, 0.73250097, 0.70593703, 0.71032012, 0.71324211, 0.71138263, 0.71921903, 0.71616417, 0.71868771, 0.72001594, 0.71988314, 0.72253948, 0.71085137, 0.71032012, 0.72214103, 0.73050869, 0.73064154, 0.73050869, 0.73037589, 0.7336964, 0.72957897, 0.72771949, 0.7283836, 0.73502457, 0.73050869, 0.71696109, 0.71390623, 0.7313056, 0.72984463, 0.7313056, 0.72240669, 0.73436046, 0.73874354, 0.73329794, 0.73449332, 0.73303229, 0.73555583, 0.73409486, 0.74086863, 0.72718823, 0.73157126, 0.7285164, 0.73050869, 0.73037589, 0.73462611, 0.7071324, 0.70939034, 0.72612566, 0.72811794, 0.73050869, 0.72732103, 0.70606989, 0.71550006, 0.71430469, 0.7121796, 0.72214103, 0.71855491]

xl = ['[1]', '[2]', '[5]', '[5, 5]', '[20, 20]', '[50, 50]']
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
