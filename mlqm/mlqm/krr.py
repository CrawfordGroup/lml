import numpy as np
from numpy import linalg as la
import copy
import json
from . import datahelper

def train(ds, **kwargs):
# {{{
    """
    Pass in a Dataset with grand representations and values, and a list of training points
    Training representations and values will be pulled from ds.grand
    Hyperparameters and coefficients will be determined and saved to ds.data
    Returns trained dataset and the average of the training values
    """

    t_REPS = [ds.grand['representations'][tr] for tr in ds.data['trainers']]
    t_VALS = [ds.grand['values'][tr] for tr in ds.data['trainers']]

    # For convenience, set the mean of the training values to 0
    t_AVG = np.mean(t_VALS)
    t_VALS = np.subtract(t_VALS,t_AVG)

    # model determination (`s` and `l` hypers, then `a` coefficients)
    # {{{
    # train the hypers
    if ds.data['hypers']:
        print("Loading hyperparameters from Dataset.")
        s = ds.data['s']
        l = ds.data['l']
    else:
        if 'k' in kwargs:
            k = kwargs['k']
        else:
            k = ds.setup['M']
        s, l = find_hypers(t_VALS,t_REPS,k)
        ds.data['hypers'] = True
        ds.data['s'] = s
        ds.data['l'] = l

    # train for alpha
    if ds.data['a']:
        print("Loading coefficients from Dataset.") 
        alpha = np.asarray(ds.data['a'])
    else:
        print("Model training using s = {} and l = {} . . .".format(s,l))
        alpha = train_a(t_REPS,t_VALS,s,l)
        ds.data['a'] = alpha.tolist()
    # }}}

    return ds, t_AVG
# }}}

def predict(ds,x):
# {{{
    '''
    for direct trained dataset use
    predict answers "Y" across entire set "x"
    given Dataset with a (regression coefficient 
    vector), and hyperparameters s and l in ds.data dict
    also given the representations of the set x
    returns answers Y 
    '''
    xt = [ds.grand['representations'][tr] for tr in ds.data['trainers']]

    # make the full k matrix [k(train,predict)]
    k = np.zeros((len(x),ds.setup["M"]))
    for i in range(0,len(x)):
        for j in range(0,ds.setup["M"]):
            k[i][j] = make_k(xt[j],x[i],ds.data['s'])

    # predict energy across the range
    Y = [] # 
    for i in range(0,len(x)):
        Y.append(np.dot(ds.data['a'],k[i]))
    
    return Y
# }}}

def pred(x,xt,a,s=0.05,l=0):
# {{{
    '''
    For in-KRR use
    predict answers "Y" across entire PES
    given x (full set of TATRs), xt (training TATRs), and a (alpha)
    optional kernel width s, regularization term l
    return Y 
    '''
    N = len(x)
    M = len(xt)

    # make the full k matrix [k(train,predict)]
    k = np.zeros((N,M))
    for i in range(0,N):
        for j in range(0,M):
            k[i][j] = make_k(xt[j],x[i],s)

    # predict energy for the entire PES
    Y = [] # 
    for i in range(0,N):
        Y.append(np.dot(a,k[i]))
    
    return Y
# }}}

def train_a(x,y,s=0.05,l=0):
# {{{
    '''
    optimize parameter(s) "a" by solving linear equations
    given x (training reps) and y (training energies)
    optional kernel width s, regularization term la
    return a
    '''
    # make the trainer-k matrix
    M = len(x)
    k = np.zeros((M,M))
    for i in range(0,M):
        for j in range(0,M):
            k[i][j] = make_k(x[i],x[j],s)

    # solve for a
    a = solve_alpha(k,y,l)

    return a
# }}}

def solve_alpha(K,Y,l=0):
# {{{
    '''
    solve the kernel ridge regression expression for alpha
    give Kernel matrix and training matrix
    optional lambda parameter
    '''
    return la.solve((K-l*np.eye(len(Y))),Y)
# }}}

def make_k(rep1, rep2, s):
# {{{
    '''
    return a gaussian kernel from two reps
    '''
    rep1 = np.asarray(rep1)
    rep2 = np.asarray(rep2)
    tmp = la.norm(rep1 - rep2)
    return np.exp(-1 * tmp**2 / (2.0*s**2))
# }}}

def find_hypers(t_y,t_reps,k):
# {{{
    s, l = log_space_cv(t_y,t_reps,k)
    return s,l
# }}}

def grid_search(tr_x,val_x,tr_y,val_y):
# {{{
    mse = 1E9
    s_list = np.logspace(-5,8,num=16)
    l_list = np.logspace(-8,-1,num=16)
    for s in s_list:
        for l in l_list:
            a = train_a(tr_x,tr_y,s,l)
            y_p = pred(val_x,tr_x,a,s,l)
            new_mse = abs(datahelper.reg_l2(val_y,y_p,l,a))
            if new_mse <= mse:
                s_f = s
                l_f = l
                mse = new_mse
    return s_f, l_f, mse
# }}}

def log_space_cv(y_data,x_data,k=5):
# {{{
    '''
    Cross-validation by searching over a log space for hyperparameters. 
    Optimize hyperparameter(s) for one fold, push that into the next, 
    optimize, and repeat. 
    '''
    print("Cross-validating in {} folds . . .".format(k))

    # split full data into k folds
    y_data = np.array_split(y_data,k) 
    x_data = np.array_split(x_data,k) 

    # hold hyperparameters and errors for averaging
    s_list = []
    l_list = []
    mse_list = []

    for val in range(0,k): # loop over validation sets
        print("Fold {} . . .".format(val+1))
        tr_y = copy.deepcopy(y_data) # copy full data set
        tr_x = copy.deepcopy(x_data)
        val_y = tr_y.pop(val) # pop out the validation set
        val_x = tr_x.pop(val)
        tr_y = np.concatenate(tr_y) # re-form training set
        tr_x = np.concatenate(tr_x) 
        s, l, mse = grid_search(tr_x,val_x,tr_y,val_y)
        s_list.append(s)
        l_list.append(l)
        mse_list.append(mse)
    s = np.mean(s_list)
    l = np.mean(l_list)
    return s,l 
# }}}
