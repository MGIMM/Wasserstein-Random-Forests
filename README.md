# Wasserstein-Random-Forests
A Random Forests-based conditional distribution estimator.

## Installation

WassersteinRandomForest mainly depends on NumPy and Cython. So make sure these dependencies are installed using pip:

```
pip3 install setuptools numpy cython
```

The rest of the dependences will be automatically installed when building.

Install from source:

```
git clone https://github.com/MGIMM/Wasserstein-Random-Forests.git
```
and then install with `pip`

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
X = np.random.uniform(0,1,(10000,4))
Y = np.array([np.random.normal(2.*X[i,0],2*X[1],1) for i in range(10000)])

reg = WassersteinRandomForest(nodesize = 2, 
                              bootstrap = False,
                              subsample = 0.01,
                              n_estimators = 500,
                              mtry = 4,
                              #n_jobs = 2, #currently unavailable
                              p = 2 #order of Wasserstein distance)
reg.fit(X,Y)

# predict conditional expectation on a new point

ref.predict(X = np.random.uniform(0,1,(1,4)))

# predict conditional distribution on a new point

Y,W = ref.predict_distribution(X = np.random.uniform(0,1,(1,4)))

# The final output is the weighted empirical measure Y*W.

```

## Example

A generic example with visulization can be found in `./test/test.py`.
