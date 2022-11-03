# pulseanalysis
Code for analyzing pulse shapes from MKID data

## Setting up environment
Install Conda and create a new environment with Python 3.7 and activate it
```bash
conda create --name <env_name> python=3.7
conda activate <env_name>
```

Download and install `pulseanalysis`
```bash
cd <directory>
git clone https://github.com/millerja14/pulseanalysis.git
pip install -e pulseanalysis
```

Download and install `mkidcalculator` v0.10.0
```bash
cd <directory>
git clone --branch 0.10.0 https://github.com/zobristnicholas/mkidcalculator.git
pip install -e mkidcalculator
```

Install `ipython` and open a session
```bash
conda install -c anaconda ipython
ipython
```

Within `ipython`, import `pulseanalysis` modules
```python
import pulseanalysis.data as data
import pulseanalysis.hist as hist
import pulseanalysis.pca as pca
```
You should be ready to start using `pulseanalysis`

## Importing data
Data that is in a format native to the `mkidcalculator` package can be imported using `pulseanalysis/data.py`.
```python
# Import traces
traces = data.loadTraces()
```
It may be easier for some users, however, to manually define an `NxM` array containing `N` traces of length `M` and pass this into the analysis functions.

> If using `data.py`, the function `loadTraces()` must be modified to match the location of the `loop` data and to remove/modify the outlier indices.

## Analyzing data
The `master` branch of `pulseanalysis` is designed to measure energy resolution of a detector using data measured from an Fe55 source.

First, lets visualize the separability of the traces in two dimensions.
```python
# get the coordinate of each trace in the 2D PCA basis
points, labels = generateScatter_labeled(2, traces=traces)
```

The `points` array contains the 2D coordinate of each trace in the space of the first two principal components. The `labels` array contains a values 0 or 1 for each trace, which is a guess of which energy peak it belongs to based on a non-pca method (integration).

We can visualize this data.
```python
# plot each trace on a 2D scatter plot
plot2DScatter(points=points, labels=labels)
```
Plots are saved as `.pdf` images in the current directory.

Next, we will find the direction in this low-dimensional space that represents changing energy. This will be the direction that maximizes separability (a.k.a. FWHM or energy resolution) when the data is projected on to it. This direction for your data in 2D is shown as an arrow on the plot generated above.

To scale to a high-dimensional subspace and potentially find a direction with better separability, we need to project the traces onto more principal components. We can see the effect of adding principal components one-by-one.
```python
# calculate energy resolution as a function of pca dimension
cc_results = componentContribution_compare(n1=3, n2=80, traces=traces, drawPlot=False)
# plot results
plot_componentContribution_compare(componentContribution_results=cc_results, id="my_first_test")
```

Again, the plot will be saved in your current directory.

In this example, `n1` is the number of pca dimensions to explore using "full" optimization and `n2` is the number of dimensions to explore using "recursive" optimization. Full optimization is one n-dimensional optimization while recursive optimization is n 1-dimensional optimizations and thus recursive optimization is significantly faster.

Finally, we can select which PCA components we want to include in our final calculation of energies. If using the recursive optimizer, it is not necessary to pick and choose components because the optization is fast. We can simply use the first 50 or the first 80.
```python

```

If using the full optimizer however, choosing more than five or so components will take a significant amount of time to run. For this reason, you have the option to pick and choose which components to use.
```python
# use recursive optimization to get most impactful components (largest change in FWHM)
# one can also just select these components manually
comp_list = getImpactfulComponents_cartesian(n=5, dim=10, points=None, labels=None, seed=1234)
```
