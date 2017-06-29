# Jet classification 
Modules for the jet classification work.

# Running examples

  * Jet images and N-subjettiness plot
```shell
python modules/get_data.py data 250-300 
```
![](plots/signal_image.png)
![](plots/backgr_image.png)
![](plots/nsubjettiness.png)

  * Principal components analysis
```shell
python modules/pca_analysis.py data 250-300
```
![](plots/pca_components.png)

  * ROC curve
```shell
python modules/plot_roc_curve.py data 250-300
```
![](plots/roc_curve.png)
