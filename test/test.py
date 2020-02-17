from WassersteinRandomForests import WassersteinRandomForest
import matplotlib.pyplot as plt
import numpy as np

N_total =51000
N_train = 50000
X = np.random.uniform(0,1,(N_total,4))
#X = np.random.normal(0,1,(N_total,10))
def obj_func(x):
    """
    conditional expectation
    """
    # x0_tilde = 2.*(x[0] - 0.5)
    # x1_tilde = 2.*(x[1] - 0.5)
    # return x0_tilde**2 + x[3]*x[4]
    # c = 0
    # for i in range(3):
    #     #c *= x[i]*np.sin(i)
    #     c += x[i]*np.sin(i)
    return 10*x[1] + x[2]
    return c
def obj_func2(x):
    """
    conditional variance
    """
    #return np.abs(np.sin(x[0])+x[1])*1.5
    #return np.abs(x[0]+x[1])*1.5
    return np.max([x[3]*9., 0.2])
    #return 0.5

def obj_func3(x):
    """
    conditional expectation
    """
    #return x[1]+2.*x[2] +2. +np.sin(2.*x[0])
    c = 0
    for i in range(4,7):
        #c *= x[i]*np.sin(i)
        c += x[i]*np.cos(i)
    #return x[1] + x[2]
    return c


#Y = np.random.normal(0,1.,N_total) + np.apply_along_axis(obj_func,1,X)
Y = np.zeros(N_total)
for i in range(N_total):
    if np.random.rand()<0.5:
        Y[i] = np.random.normal(obj_func(X[i]),np.sqrt(obj_func2(X[i])),1)
    else:
        #Y[i] = np.random.normal(obj_func(X[i]),np.sqrt(obj_func2(X[i])),1)
        # Y[i] = np.random.normal(-0.05*obj_func2(X[i]),1,1)
        #Y[i] = np.random.normal(-1.5*obj_func2(X[i]),1,1)
        #Y[i] = np.random.normal(obj_func3(X[i]),1,1)

        Y[i] = np.random.normal(-1,1,1)
        #Y[i] = np.random.normal(obj_func(X[i]),np.sqrt(obj_func2(X[i])),1)

reg = WassersteinRandomForest(nodesize = 2,
                             bootstrap = False,
                             subsample = 0.005,
                             n_estimators = 500,
                             mtry = 4,
                             #n_jobs = 1,
                             p = 2)
#reg = DecisionTree()
reg.fit(X[:N_train],Y[:N_train])

from sklearn.metrics import mean_squared_error, r2_score
predlist = reg.predict(X[N_train:])
print("Wasserstein RF")
print("R2:",r2_score(np.apply_along_axis(obj_func,1,X[N_train:]),predlist))
print("MSE:",
        np.sqrt(mean_squared_error(np.apply_along_axis(obj_func,1,X[N_train:]),predlist))
     )
#print("Memory usage:", process.memory_info().rss/1024/1024,"MB")



from seaborn import kdeplot

plt.figure(figsize = (20,15))
for IndexPlot in range(9):
    plt.subplot(int("33"+str(IndexPlot+1)))
    TestIndex = np.random.choice(range(N_total-N_train),1)[0]
    #TestIndex = np.random.choice(range(N_train),1)[0]
    #plt.hist(Y, density = True, color = "orange",alpha = 0.3, bins = 20, label="Y")
    kdeplot(Y, label="kde-Y", color = "darkorange")
    Y,W = reg.predict_distribution(X[-TestIndex])
    kdeplot(np.random.choice(a = Y,p = W,size = 1000), label="kde-pred", color = "black")
    plt.hist(Y,weights=W, bins = 20, color = "grey", density = True,alpha = 0.5, label="pred")
    #ref_sample = np.random.normal(obj_func(X[-TestIndex]),np.sqrt(obj_func2(X[-TestIndex])),2000)
    ref_sample = np.zeros(2000)
    for i in range(2000):
        if np.random.rand()<0.5:
            ref_sample[i] = np.random.normal(obj_func(X[-TestIndex]),np.sqrt(obj_func2(X[-TestIndex])),1)
        else:
            #ref_sample[i] = np.random.normal(-1.5*obj_func2(X[TestIndex]),1,1)
            # ref_sample[i] = np.random.normal(obj_func3(X[TestIndex]),1,1)

            #ref_sample[i] = np.random.normal(-1,1,1)
            ref_sample[i] = np.random.normal(obj_func(X[-TestIndex]),np.sqrt(obj_func2(X[-TestIndex])),1)

    plt.hist(ref_sample,
             density = True, color = "darkred",alpha = 0.3, bins = 20, label="ref")
    kdeplot(ref_sample, label = "kde-ref", color = "darkred")
    plt.grid(linestyle = "-.",color="lightgrey")
    plt.legend()
plt.show()
