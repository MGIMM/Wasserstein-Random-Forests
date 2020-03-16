
# model 1

```
N_total =10000
N_train = 5000

X = np.random.uniform(0,1,(N_total,50))
def obj_func(x):
    """
    conditional expectation
    """
    return 10.*x[1]*x[3]**2 + x[2] + np.exp(x[3]-2*x[0]+np.sin(x[1]**3))
def obj_func2(x):
    """
    conditional variance
    """
    return np.max([(x[0]*x[1]+np.cos(x[2]*x[3]**2))*2.5, 0.2])
    


Y = np.zeros(N_total)
for i in range(N_total):
    if np.random.rand()<0.5:
        Y[i] = np.random.normal(obj_func(X[i]),np.sqrt(obj_func2(X[i])),1) 
    else:
        Y[i] = np.random.normal(obj_func(X[i]),np.sqrt(obj_func2(X[i])),1) 

```

# Breiman's RF

## parameter:

```
reg = RandomForest(nodesize = 5,
                   bootstrap = True,
                   subsample = 5000,
                   n_estimators = 100,
                   mtry = 20,
                   #n_jobs = 1,
                   p = 2)
```
## results

R2: 0.957001750809336

MSE: 0.5070201440616019

average Wp distance: 0.6478838876699471

average Wp distance with Y (i.e., no estimation is made): 2.555506060589508

ideal average Wp distance 0.05320298591960361

# Wasserstein RF

## parameter:

```
reg = WassersteinRandomForest(nodesize = 5,
                             bootstrap = True,
                             subsample = 5000,
                             n_estimators = 100,
                             mtry = 20,
                             #n_jobs = 1,
                             p = 2)
```
## results

R2: 0.9600643112379628

MSE: 0.504617881459418

average Wp distance: 0.6086505446860154

average Wp distance with Y (i.e., no estimation is made): 2.18645137665258

ideal average Wp distance 0.052722513866543926
