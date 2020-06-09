# Wasserstein-Random-Forests
A Random Forests-based conditional distribution estimator:

(X_i,Y_i; 1<= i <= n) + WRF estimate L(Y | X = x) for each x in Supp(X).

<img src="fig/multimodal.png" height="500" />


## Installation and Dependencies

### Dependencies

**WassersteinRandomForest** is based on `NumPy` and `Cython`. 
So, make sure these packages are installed. For example, you can install them with `pip`:

```
pip3 install setuptools numpy cython
```

The rest of the dependences (currently, `tqdm`) will be automatically installed
when building. For `Windows 10` user, you will need to make sure `Visual
Studio` is installed properly and all the `c++` dependencies are available.
For `linux` user, make sure `python3-dev` and `gcc` are installed properly.

In addition, you also need:

* `jupyter` for jupyter notebooks;

* `seaborn` and `matplotlib` in order to be able to test all the visualizations;

* `pickle` to save and load models;

* `POT` to compare Wasserstein-1d distance.

Finally, it is recommend to use `Python 3.7` from [Anaconda](https://www.anaconda.com/) distribution. All the codes for the article are tested on Ubuntu 20.04.


### Install from source:

Install with `pip`:

```
cd Wasserstein-Random-Forests
```
and
```
pip install .
```



## Usage


```python
import numpy as np
from WassersteinRandomForests import WassersteinRandomForest  

# generate synthetic data
X = np.random.uniform(0,1,(1000,50))
Y = np.array([np.random.normal(2.*X[i,0],2.*X[i,1],1) for i in
range(1000)]).reshape(1000,)

reg = WassersteinRandomForest(nodesize = 5, # upper bound of the leaves 
                              bootstrap = True, # bootstrap 
                              subsample = 500, # when subsample <= 1, subsample is the resampling rate; when subsample >1, sumbsample = number of sample points for each tree 
                              n_estimators = 100, # number of decision trees
                              mtry = 40, # max features used for splitting
                              p = 2, # order of Wasserstein distance
                              interpretation = "intra", # methods in ["inter","intra"], note that when "intra" is selected, p is automatically selected as 2. 
                              ) 
reg.fit(X,Y)

# predict conditional expectation on a new point

reg.predict(X = np.random.uniform(0,1,(1,50)))

# predict conditional distribution on a new point

Y,W = reg.predict_distribution(X = np.random.uniform(0,1,(1,50)))

# The final output is the weighted empirical measure Y*W.

```

To reproduce the results and visualizations in the main text, a `jupyter notebook` and a `.html` file are provided
in `./notebooks/Main_text.ipynb`.

## Remarks on training time 

Currently, the package only provides an `Cython` accelaration for the splitting
mechanism. The tree construction and prediction are still implemented with raw
`python` code.
All the average running time on a single core of the CPU: Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz can be found in `./notebooks/Main_text.ipynb`.

## References

* Wasserstein Random Forests and Applications in Heterogenerous Treatment
  Effects. [arXiv](http://arxiv.org/abs/2006.04709)

