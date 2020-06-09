# distutils: language = c++
from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython
from libcpp.algorithm cimport sort,unique
from libcpp.vector cimport vector
from libc.math cimport ceil
from libc.math cimport pow as pow_C 

from tqdm import tqdm
#from joblib import Parallel,delayed
#import multiprocessing


@cython.boundscheck(False) 
@cython.wraparound(False)  
cdef inline double my_abs(double x) nogil:
    if x > 0:
        return x
    else:
        return -x
    
# @cython.boundscheck(False) 
# @cython.wraparound(False)  
# cdef double[:] _sort_cpp(double[:] a):
#     # a must be c continuous (enforced with [::1])
#     sort(&a[0], (&a[0]) + a.shape[0])
#     return a 

@cython.boundscheck(False) 
@cython.wraparound(False)    
cdef inline double _Wp(vector[double] P, vector[double] Q, int p = 2) nogil:
    cdef int N1_int = P.size()
    cdef int N2_int = Q.size()
    #cdef int N1_int = P.shape[0]
    #cdef int N2_int = Q.shape[0]
    cdef double N1 = N1_int
    cdef double N2 = N2_int 
    cdef int j
    cdef double dist = 0
    cdef int i
    cdef int N_max 
    cdef double inverse_N1 = 1.0/N1
    cdef double inverse_N2 = 1.0/N2
    # cdef double[:] P_sorted = _sort_cpp(P)
    # cdef double[:] Q_sorted = _sort_cpp(Q)
    
    cdef vector[double] U1
    cdef vector[double] U2
    cdef vector[double] U
    U2.reserve(N2_int+1)
    U1.reserve(N1_int+1)
    U1.push_back(0)
    U2.push_back(0)
    
    
    sort(P.begin(),P.end())
    sort(Q.begin(),Q.end())
    
    for j in range(N1_int):
        U1.push_back(U1[j]+inverse_N1)
    
    for j in range(N2_int):
        U2.push_back(U2[j]+inverse_N2)
    # concatenate & remove duplicates & sort
    U.reserve(U1.size()+U2.size())
    U.insert( U.end(), U1.begin(), U1.end() )
    U.insert( U.end(), U2.begin(), U2.end() )
    sort(U.begin(),U.end())
    U.erase(unique(U.begin(),U.end()),U.end())
    
    N_max = U.size() - 1
    
    for i in range(N_max):
        dist += pow_C(my_abs(P[int(ceil(U[i]*(N1-1)))] - Q[int(ceil(U[i]*(N2-1)))]),p)*(U[i+1] - U[i])
        
    return pow_C(dist,1.0/float(p)) 
    #return dist

@cython.boundscheck(False) 
@cython.wraparound(False)    
cdef inline double my_w1(double x,vector[double] y) nogil:
    #cdef double N = y.size()
    cdef double result  = 0
    for i in range(y.size()):
        result += my_abs(x - y[i])
    #result /= N
    return result
    #return result
@cython.boundscheck(False) 
@cython.wraparound(False)    
cdef inline double my_mean(vector[double] x) nogil:
    cdef double result = 0
    for i in range(x.size()):
        result += x[i]
    return result/float(x.size())
# breiman's criterion0
@cython.boundscheck(False) 
@cython.wraparound(False)    
cdef inline double _calculate_L2_difference(vector[int] A,
                                     int split_direction,
                                     double split_value,
                                     double[:,:] X,
                                     double[:] Y,
                                     int p = 2) nogil:
    cdef int A_size = A.size()
    cdef int i
    cdef vector[double] Y_left
    cdef vector[double] Y_right
    cdef vector[double] Y_both
    Y_left.reserve(A_size)
    Y_right.reserve(A_size)
    Y_both.reserve(A_size)
    
    for i in range(A_size):
        Y_both.push_back(Y[A[i]])
        if X[A[i],split_direction] < split_value:
            Y_left.push_back(Y[A[i]])
            #Y_left.push_back(1)
        else:
            Y_right.push_back(Y[A[i]])
            #Y_right.push_back(1)
            
    
            
    cdef double result = 0
    cdef double mean_A = my_mean(Y_both)
    cdef double mean_L = my_mean(Y_left)
    cdef double mean_R = my_mean(Y_right)
    
    
    
    for i in range(A_size):
        result += (Y_both[i]-mean_A)*(Y_both[i]-mean_A)
    for i in range(Y_left.size()):
        result -= (Y_left[i]-mean_L)*(Y_left[i]-mean_L)
    for i in range(Y_right.size()):
        result -= (Y_right[i]-mean_R)*(Y_right[i]-mean_R)
   
    
    return result
@cython.boundscheck(False) 
@cython.wraparound(False)    
cdef inline double _calculate_Wp_difference(vector[int] A,
                                     int split_direction,
                                     double split_value,
                                     double[:,:] X,
                                     double[:] Y,
                                     int p = 2) nogil:
    cdef int A_size = A.size()
    cdef int i
    cdef vector[double] Y_left
    cdef vector[double] Y_right
    cdef vector[double] Y_both
    Y_left.reserve(A_size)
    Y_right.reserve(A_size)
    Y_both.reserve(A_size)
    
    for i in range(A_size):
        Y_both.push_back(Y[A[i]])
        if X[A[i],split_direction] < split_value:
            Y_left.push_back(Y[A[i]])
            #Y_left.push_back(1)
        else:
            Y_right.push_back(Y[A[i]])
            #Y_right.push_back(1)
            
    
            
    # cdef double result = 0
    # cdef double mean_A = my_mean(Y_both)
    # cdef double mean_L = my_mean(Y_left)
    # cdef double mean_R = my_mean(Y_right)
    # 
    # 
    # 
    # for i in range(A_size):
    #     result += (Y_both[i]-mean_A)*(Y_both[i]-mean_A)
    # for i in range(Y_left.size()):
    #     result -= (Y_left[i]-mean_L)*(Y_left[i]-mean_L)
    # for i in range(Y_right.size()):
    #     result -= (Y_right[i]-mean_R)*(Y_right[i]-mean_R)
   
    
    
    #return _Wp(P = Y_left,Q = Y_right, p = p)+_Wp(P = Y_both,Q = Y_right, p = p)+_Wp(P = Y_both,Q = Y_left, p = p)
    #return _Wp(P = Y_left,Q = Y_right, p = p)+float(Y_right.size())/float(A_size)*_Wp(P = Y_both,Q = Y_right, p = p)+float(Y_left.size())/float(A_size)*_Wp(P = Y_both,Q = Y_left, p = p)
    
    # a good choice
    return float(Y_right.size())*_Wp(P = Y_both,Q = Y_right, p = p)+float(Y_left.size())*_Wp(P = Y_both,Q = Y_left, p = p)

    #return ot.wasserstein_1d(np.array(Y_left),np.array(Y_right), p = p)
    #return result




@cython.boundscheck(False) 
@cython.wraparound(False)    
cpdef inline (vector[int],vector[int]) getALAR(vector[int] A,
                                               int split_direction,
                                               double split_value,
                                               double[:,:] X) nogil:
    
    cdef vector[int] _AL
    cdef vector[int] _AR
    cdef int A_size = A.size()
    _AL.reserve(A_size)
    _AR.reserve(A_size)
    for i in range(A_size):
        if X[A[i],split_direction] < split_value:
            _AL.push_back(A[i])
        else:
            _AR.push_back(A[i])
            
    #cdef int[:] AL = _AL.data()
    #cdef int[:] AR = _AR.data()
    
    # cdef int *AL = &_AL[0]    
    # cdef int[::1] AL_view = <int[:_AL.size()]>AL    # cast to typed memory view
    
    # cdef int *AR = &_AL[0]    
    # cdef int[::1] AR_view = <int[:_AL.size()]>AR
    # # 
    #return np.asarray(_AL,dtype=np.intc),np.asarray(_AR,dtype=np.intc)
    return _AL,_AR
    #return 0 




# Wp_split
@cython.boundscheck(False) 
@cython.wraparound(False)    
cpdef inline (int,double) Wp_split(vector[int] A,
             int[:] Mtry,
             double[:,:] X,
             double[:] Y,
             int p = 2) nogil:
    
    #cdef int min_sample_each_node = 1  
    cdef int A_size = A.size() 
    cdef int mtry = Mtry.shape[0]
    cdef int i
    cdef int j
    cdef int index_max
    cdef int _size_list = (A_size-1)*mtry
    
    cdef vector[int] direction_list 
    direction_list.reserve(_size_list)
    cdef vector[double] value_list 
    value_list.reserve(_size_list)
    
    cdef int split_direction
    
    cdef vector[double] Xtry 
    Xtry.reserve(A_size)
    
    
    for i in range(mtry):
        split_direction = Mtry[i]
        #Xtry = np.sort(list(set(X[A,:][:,split_direction])))
        for j in range(A_size):
            Xtry.push_back(X[A[j],split_direction])
        sort(Xtry.begin(),Xtry.end())
        #print("X_before",Xtry.size())
        
        Xtry.erase(unique(Xtry.begin(),Xtry.end()),Xtry.end())
        
        #print("X_after",Xtry.size())
        #Xtry = np.sort(list(set(X[A,split_direction])))
        #candidate_current_direction = (Xtry[min_sample_each_node:] + Xtry[:-min_sample_each_node])*.5
        for j in range(Xtry.size() - 1):
            direction_list.push_back(split_direction)
            value_list.push_back((Xtry[j+1]+Xtry[j])*0.5)
        Xtry.clear()
        #theta_list += [[split_direction,candidate_current_direction[i]] for i in range(len(Xtry) -min_sample_each_node)]
    
    #print("values:",value_list)
    cdef vector[double] Wp_list
    Wp_list.reserve(direction_list.size())
    for i in range(direction_list.size()):
        Wp_list.push_back(_calculate_Wp_difference(A = A,
                                                   split_direction = direction_list[i],
                                                   split_value = value_list[i],
                                                   X = X,
                                                   Y = Y,
                                                   p = p))
    index_max = my_argmax(Wp_list) 
    return direction_list[index_max], value_list[index_max] 
# L2_split
@cython.boundscheck(False) 
@cython.wraparound(False)    
cpdef inline (int,double) L2_split(vector[int] A,
             int[:] Mtry,
             double[:,:] X,
             double[:] Y,
             int p = 2) nogil:
    
    #cdef int min_sample_each_node = 1  
    cdef int A_size = A.size() 
    cdef int mtry = Mtry.shape[0]
    cdef int i
    cdef int j
    cdef int index_max
    cdef int _size_list = (A_size-1)*mtry
    
    cdef vector[int] direction_list 
    direction_list.reserve(_size_list)
    cdef vector[double] value_list 
    value_list.reserve(_size_list)
    
    cdef int split_direction
    
    cdef vector[double] Xtry 
    Xtry.reserve(A_size)
    
    
    for i in range(mtry):
        split_direction = Mtry[i]
        #Xtry = np.sort(list(set(X[A,:][:,split_direction])))
        for j in range(A_size):
            Xtry.push_back(X[A[j],split_direction])
        sort(Xtry.begin(),Xtry.end())
        #print("X_before",Xtry.size())
        Xtry.erase(unique(Xtry.begin(),Xtry.end()),Xtry.end())
        #print("X_after",Xtry.size())
        #Xtry = np.sort(list(set(X[A,split_direction])))
        #candidate_current_direction = (Xtry[min_sample_each_node:] + Xtry[:-min_sample_each_node])*.5
        for j in range(Xtry.size() - 1):
            direction_list.push_back(split_direction)
            value_list.push_back((Xtry[j+1]+Xtry[j])*0.5)
        Xtry.clear()
        #theta_list += [[split_direction,candidate_current_direction[i]] for i in range(len(Xtry) -min_sample_each_node)]
    
    #print("values:",value_list)
    cdef vector[double] Wp_list
    Wp_list.reserve(direction_list.size())
    for i in range(direction_list.size()):
        Wp_list.push_back(_calculate_L2_difference(A = A,
                                                   split_direction = direction_list[i],
                                                   split_value = value_list[i],
                                                   X = X,
                                                   Y = Y,
                                                   p = p))
    index_max = my_argmax(Wp_list) 
    return direction_list[index_max], value_list[index_max] 

@cython.boundscheck(False) 
@cython.wraparound(False)    
cdef inline int my_argmax(vector[double] x) nogil:
    cdef int i
    cdef int _i = 0
    cdef double _x = x[0]
    for i in range(x.size()):
        if x[i] > _x:
            _i = i
            _x = x[i]
    return _i

class node:
    def __init__(self, left = None, right = None, split = None, neighbours = None):
        self.left = left 
        self.right = right 
        self.split = split  
        self.neighbours = neighbours 
        
class DecisionTree:
    def __init__(self,
                 mtry = 1,
                 nodesize = 5,
                 subsample = 0.8,
                 bootstrap = True,
                 p = 2,
                 interpretation = 'inter',
                 nodes = None):
        #self.nodes = nodes # P 
        # parameters
        self.mtry = mtry
        self.nodesize = nodesize
        self.subsample = subsample 
        self.bootstrap = bootstrap 
        self.p = p 
        self.interpretation = interpretation 
        self.Y = None
        self.P = []
        self.root = None
    def fit(self,X,Y):
        self.Y = Y
        N,d = X.shape 
        if self.subsample <= 1:
            subsample_size = int(N*self.subsample)
        elif self.subsample > 1:
            subsample_size = int(self.subsample)
        else:
            subsample_size = 100
            print("Wrong subsample is given, used subsample = 100 as default.")
        S_b = np.random.choice(range(N), subsample_size, replace  = self.bootstrap)
        self.root = node(neighbours = S_b)
        self.P +=[self.root] 
        if self.interpretation == 'inter':
            while self.P:
                # A is current node
                A = self.P[0] 
                if len(A.neighbours) < self.nodesize or len(set(Y[A.neighbours])) == 1:
                    del self.P[0]
                else:
                    Mtry = np.random.choice(range(d),self.mtry,replace = False)
                    Mtry = np.array(Mtry,dtype = np.intc)
                    theta_star = Wp_split(A.neighbours,Mtry,X,Y,self.p)
                    A.split = theta_star
                    #theta_star_list += [theta_star] 
                    AL,AR = getALAR(A.neighbours,theta_star[0],theta_star[1],X)
                    del self.P[0]
                    A.left = node(neighbours = AL)
                    A.right = node(neighbours = AR)
                    self.P += [A.left,A.right]
        elif self.interpretation == 'intra':
            while self.P:
                # A is current node
                A = self.P[0] 
                if len(A.neighbours) < self.nodesize or len(set(Y[A.neighbours])) == 1:
                    del self.P[0]
                else:
                    Mtry = np.random.choice(range(d),self.mtry,replace = False)
                    Mtry = np.array(Mtry,dtype = np.intc)
                    theta_star = L2_split(A.neighbours,Mtry,X,Y,self.p)
                    A.split = theta_star
                    #theta_star_list += [theta_star] 
                    AL,AR = getALAR(A.neighbours,theta_star[0],theta_star[1],X)
                    del self.P[0]
                    A.left = node(neighbours = AL)
                    A.right = node(neighbours = AR)
                    self.P += [A.left,A.right]
        
    def _predict(self,x):
        current_node = self.root
        while current_node.split:
            direction,value = current_node.split
            if x[direction]<value:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return np.mean(self.Y[current_node.neighbours])
    def predict_distribution(self,x):
        current_node = self.root
        while current_node.split:
            direction,value = current_node.split
            if x[direction]<value:
                current_node = current_node.left
            else:
                current_node = current_node.right
        N_final = self.Y.shape[0]
        empirical_measure = np.zeros(N_final)
        for i in current_node.neighbours:
            empirical_measure[i] += 1.
        return empirical_measure/float(len(current_node.neighbours))
            
    def predict(self,x):
        return np.apply_along_axis(lambda x : self._predict(x),1,x)
        
class WassersteinRandomForest:
    def __init__(self,
                 mtry = 1,
                 nodesize = 5,
                 subsample = 0.1,
                 bootstrap = False,
                 n_estimators = 10,
                 #n_jobs = 1,
                 p = 2,
                 interpretation = 'inter'):
        # parameters
        self.mtry = mtry
        self.nodesize = nodesize
        self.subsample = subsample 
        self.bootstrap = bootstrap 
        self.n_estimators = n_estimators
        self.interpretation = interpretation 
        #self.n_jobs = n_jobs
        self.n_jobs = 1 
        self.p = p
        #self.ListLearners = None 
        self.ListLearners = []
        self.Y = None
        
    def reset_random_state(self):
        f = open("/dev/random","rb")
        rnd_str = f.read(4)
        rnd_int = int.from_bytes(rnd_str, byteorder = 'big')
        np.random.seed(rnd_int)

    def _fit(self,X,Y):
        # if self.n_jobs > 1:
        #     self.reset_random_state()
        BaseLearner = DecisionTree(mtry = self.mtry,
                                   nodesize = self.nodesize,
                                   subsample = self.subsample,
                                   bootstrap = self.bootstrap,
                                   p = self.p,
                                   interpretation = self.interpretation)
        BaseLearner.fit(X,Y)
        return BaseLearner

    def fit(self,X,Y):
        self.Y = Y
        if self.n_jobs ==1:
            #self.ListLearners = []
            for i in tqdm(range(self.n_estimators)):
                self.ListLearners += [self._fit(X,Y)]
        #else:
            
            #ListLearners = Parallel(n_jobs=self.n_jobs)(delayed(self._fit)(X,Y) for i in tqdm(range(self.n_estimators)))

            # results =\
            # Parallel(n_jobs=self.n_jobs)(delayed(self._fit) (X=X,Y=Y) for i in tqdm(range(self.n_estimators)))
# 
            # self.ListLearners = results 
    # def fit(self,X,Y):
    #     self.Y = Y
    #     for i in tqdm(range(self.n_estimators)):
    #         #BaseLearner = DecisionTree(mtry = self.mtry,
    #         #                           nodesize = self.nodesize,
    #         #                           subsample = self.subsample,
    #         #                           bootstrap = self.bootstrap,
    #         #                           p = self.p)
    #         #BaseLearner.fit(X,Y)
    #         self.ListLearners += [self._fit(X,Y)]
    def predict(self,x):
        prediction = np.zeros(x.shape[0])
        for i in range(self.n_estimators):
            prediction += self.ListLearners[i].predict(x)
        prediction /= float(self.n_estimators)
        return prediction
    def predict_distribution(self,x):
        empirical_measure = np.zeros(len(self.Y))
        for i in range(self.n_estimators):
            current_empirical_measure= self.ListLearners[i].predict_distribution(x)
            empirical_measure += current_empirical_measure
        return self.Y,empirical_measure/float(self.n_estimators)


# Multivariate WRF

@cython.boundscheck(False) 
@cython.wraparound(False)    
@cython.cdivision(True)
cdef inline double my_mean_multi(double[:] x) nogil:
    cdef double result = 0
    for i in range(x.shape[0]):
        result += x[i]
    return result/float(x.shape[0])
# breiman's criterion

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double _calculate_L_intra_2_multi_difference(int[:] A,
                                     int split_direction,
                                     double split_value,
                                     double[:,:] X,
                                     double[:,:] Y
                                     ):
    cdef int A_size = A.shape[0]
    cdef int i,k
    cdef int left_size = 0
    cdef int dimY = Y.shape[1]

    cdef double[:,:] Y_both = np.zeros((A_size,dimY))
    cdef double[:,:] Y_left = np.zeros((A_size,dimY))
    cdef double[:,:] Y_right = np.zeros((A_size,dimY))
    cdef int index_left = 0
    cdef int index_right = 0

    cdef double result = 0
    cdef vector[double] mean_A
    cdef vector[double] mean_L
    cdef vector[double] mean_R
    mean_A.reserve(dimY)
    mean_L.reserve(dimY)
    mean_R.reserve(dimY)

    with nogil:
        for i in range(A_size):
            if X[A[i],split_direction] < split_value:
                left_size += 1
        for i in range(A_size):
            for k in range(dimY):
                Y_both[i,k] = Y[A[i],k]
            if X[A[i],split_direction] < split_value:
                for k in range(dimY):
                    Y_left[index_left,k] = Y[A[i],k]
                index_left += 1
            else:
                for k in range(dimY):
                    Y_right[index_right,k] = Y[A[i],k]
                index_right += 1



        for k in range(dimY):
            mean_A.push_back(my_mean_multi(Y_both[:,k]))
            mean_L.push_back(my_mean_multi(Y_left[:index_left+1,k]))
            mean_R.push_back(my_mean_multi(Y_right[:index_right+1,k]))
        #
        # #cdef double result_L = 0
        # #cdef double result_R = 0
        # #cdef double f_A_size = Y_both.size()
        # #cdef double f_L_size = Y_left.size()
        # #cdef double f_R_size = Y_right.size()
        # #f_L_size *=f_L_size
        # #f_R_size *=f_R_size
        #
        #
        for i in range(A_size):
            for k in range(dimY):
                result += (Y_both[i,k]-mean_A[k])*(Y_both[i,k]-mean_A[k])
            #result += my_w1(Y_both[i],Y_both)
        for i in range(index_left+1):
            for k in range(dimY):
                result -= (Y_left[i,k]-mean_L[k])*(Y_left[i,k]-mean_L[k])
            #result_L += (Y_left[i]-mean_L)*(Y_left[i]-mean_L)
            #result -= my_w1(Y_left[i],Y_left)
        for i in range(index_right+1):
            for k in range(dimY):
                result -= (Y_right[i,k]-mean_R[k])*(Y_right[i,k]-mean_R[k])
        return result

    #return ot.wasserstein_1d(np.array(Y_left),np.array(Y_right), p = p)
    #return result




@cython.boundscheck(False)
@cython.wraparound(False)
def getALAR_multi(int[:] A,
            int split_direction,
            double split_value,
            double[:,:] X):

    cdef vector[int] _AL
    cdef vector[int] _AR
    cdef int A_size = A.shape[0]
    _AL.reserve(A_size)
    _AR.reserve(A_size)
    with nogil:
        for i in range(A_size):
            if X[A[i],split_direction] < split_value:
                _AL.push_back(A[i])
            else:
                _AR.push_back(A[i])

    #cdef int[:] AL = _AL.data()
    #cdef int[:] AR = _AR.data()

    # cdef int *AL = &_AL[0]
    # cdef int[::1] AL_view = <int[:_AL.size()]>AL    # cast to typed memory view

    # cdef int *AR = &_AL[0]
    # cdef int[::1] AR_view = <int[:_AL.size()]>AR
    # #
    return np.asarray(_AL,dtype=np.intc),np.asarray(_AR,dtype=np.intc)
    #return _AL,_AR
    #return 0




# Wp_split
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline (int,double) Multi_Breiman_split(int[:] A,
                                        int[:] Mtry,
                                        double[:,:] X,
                                        double[:,:] Y
                                        ):

    #cdef int min_sample_each_node = 1
    cdef int A_size = A.shape[0]
    cdef int mtry = Mtry.shape[0]
    cdef int i
    cdef int j
    cdef int index_max
    cdef int _size_list = (A_size-1)*mtry

    cdef vector[int] direction_list
    direction_list.reserve(_size_list)
    cdef vector[double] value_list
    value_list.reserve(_size_list)

    cdef int split_direction

    cdef vector[double] Xtry
    Xtry.reserve(A_size)

    cdef vector[double] criteria_list
    criteria_list.reserve(direction_list.size())

    with nogil:
        for i in range(mtry):
            split_direction = Mtry[i]
            #Xtry = np.sort(list(set(X[A,:][:,split_direction])))
            for j in range(A_size):
                Xtry.push_back(X[A[j],split_direction])
            sort(Xtry.begin(),Xtry.end())
            #print("X_before",Xtry.size())
            Xtry.erase(unique(Xtry.begin(),Xtry.end()),Xtry.end())
            #print("X_after",Xtry.size())
            #Xtry = np.sort(list(set(X[A,split_direction])))
            #candidate_current_direction = (Xtry[min_sample_each_node:] + Xtry[:-min_sample_each_node])*.5
            for j in range(Xtry.size() - 1):
                direction_list.push_back(split_direction)
                value_list.push_back((Xtry[j+1]+Xtry[j])*0.5)
            Xtry.clear()
            #theta_list += [[split_direction,candidate_current_direction[i]] for i in range(len(Xtry) -min_sample_each_node)]

    #print("values:",value_list)
    for i in range(direction_list.size()):
        criteria_list.push_back(_calculate_L_intra_2_multi_difference(A = A,
                                                        split_direction = direction_list[i],
                                                        split_value = value_list[i],
                                                        X = X,
                                                        Y = Y
                                                        ))
    index_max = my_argmax(criteria_list)
    return direction_list[index_max], value_list[index_max]

class DecisionTree_multi:
    def __init__(self,
                 mtry = 1,
                 nodesize = 5,
                 subsample = 0.8,
                 bootstrap = True,
                 nodes = None):
        #self.nodes = nodes # P
        # parameters
        self.mtry = mtry
        self.nodesize = nodesize
        self.subsample = subsample
        self.bootstrap = bootstrap
        self.Y = None
        self.P = []
        self.root = None
    def fit(self,X,Y):
        self.Y = Y
        N,d = X.shape
        dim_Y = self.Y.shape[1]
        if self.subsample <= 1:
            subsample_size = int(N*self.subsample)
        elif self.subsample > 1:
            subsample_size = int(self.subsample)
        else:
            subsample_size = 100
            print("Wrong subsample is given, used subsample = 100 as default.")
        S_b = np.random.choice(range(N), subsample_size, replace  = self.bootstrap)
        self.root = node(neighbours = np.array(S_b,dtype = np.intc))
        self.P +=[self.root]
        while self.P:
            # A is current node
            A = self.P[0]
            if len(A.neighbours) < self.nodesize or len(np.unique(Y[A.neighbours])) == dim_Y:
            #if len(A.neighbours) < self.nodesize:
                del self.P[0]
            else:
                Mtry = np.random.choice(range(d),self.mtry,replace = False)
                Mtry = np.array(Mtry,dtype = np.intc)
                theta_star = Multi_Breiman_split(A.neighbours,Mtry,X,Y)
                A.split = theta_star
                #theta_star_list += [theta_star]
                AL,AR = getALAR_multi(np.asarray(A.neighbours,dtype = np.intc),theta_star[0],theta_star[1],X)
                del self.P[0]
                A.left = node(neighbours = AL)
                A.right = node(neighbours = AR)
                self.P += [A.left,A.right]

    def _predict(self,x):
        current_node = self.root
        while current_node.split:
            direction,value = current_node.split
            if x[direction]<value:
                current_node = current_node.left
            else:
                current_node = current_node.right
        #return np.mean(self.Y[current_node.neighbours])
        return np.apply_along_axis(np.mean,0,self.Y[current_node.neighbours])
    def predict_distribution(self,x):
        current_node = self.root
        while current_node.split:
            direction,value = current_node.split
            if x[direction]<value:
                current_node = current_node.left
            else:
                current_node = current_node.right
        N_final = self.Y.shape[0]
        empirical_measure = np.zeros(N_final)
        for i in current_node.neighbours:
            empirical_measure[i] += 1.
        return empirical_measure/float(len(current_node.neighbours))

    def predict(self,x):
        return np.apply_along_axis(lambda x : self._predict(x),1,x)

class WassersteinRandomForest_multi:
    def __init__(self,
                 mtry = 1,
                 nodesize = 5,
                 subsample = 0.1,
                 bootstrap = False,
                 n_estimators = 10,
                 #n_jobs = 1,
                 ):
        # parameters
        self.mtry = mtry
        self.nodesize = nodesize
        self.subsample = subsample
        self.bootstrap = bootstrap
        self.n_estimators = n_estimators
        #self.n_jobs = n_jobs
        self.n_jobs = 1
        #self.p = p
        #self.ListLearners = None
        self.ListLearners = []
        self.Y = None

    # def reset_random_state(self):
    #     f = open("/dev/random","rb")
    #     rnd_str = f.read(4)
    #     rnd_int = int.from_bytes(rnd_str, byteorder = 'big')
    #     np.random.seed(rnd_int)

    def _fit(self,X,Y):
        # if self.n_jobs > 1:
        #     self.reset_random_state()
        BaseLearner = DecisionTree_multi(mtry = self.mtry,
                                   nodesize = self.nodesize,
                                   subsample = self.subsample,
                                   bootstrap = self.bootstrap
                                   )
        BaseLearner.fit(X,Y)
        return BaseLearner

    def fit(self,X,Y):
        self.Y = Y
        if self.n_jobs ==1:
            #self.ListLearners = []
            for i in tqdm(range(self.n_estimators)):
                self.ListLearners += [self._fit(X,self.Y)]
        # else:
        #     #ListLearners = Parallel(n_jobs=self.n_jobs)(delayed(self._fit)(X,Y) for i in tqdm(range(self.n_estimators)))

        #     results =\
        #     Parallel(n_jobs=self.n_jobs)(delayed(self._fit) (X=X,Y=Y) for i in tqdm(range(self.n_estimators)))

        #     self.ListLearners = results
    # def fit(self,X,Y):
    #     self.Y = Y
    #     for i in tqdm(range(self.n_estimators)):
    #         #BaseLearner = DecisionTree(mtry = self.mtry,
    #         #                           nodesize = self.nodesize,
    #         #                           subsample = self.subsample,
    #         #                           bootstrap = self.bootstrap,
    #         #                           p = self.p)
    #         #BaseLearner.fit(X,Y)
    #         self.ListLearners += [self._fit(X,Y)]
    def predict(self,x):
        prediction = np.zeros((x.shape[0],self.Y.shape[1]))
        for i in range(self.n_estimators):
            prediction += self.ListLearners[i].predict(x)
        prediction /= float(self.n_estimators)
        return prediction
    def predict_distribution(self,x):
        empirical_measure = np.zeros(len(self.Y))
        for i in range(self.n_estimators):
            current_empirical_measure= self.ListLearners[i].predict_distribution(x)
            empirical_measure += current_empirical_measure
        return self.Y,empirical_measure/float(self.n_estimators)


