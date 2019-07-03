import numpy as np
from numpy import linalg as la
import copy
import json
from . import datahelper

def krr(ds, trainers, validators, **kwargs):
# {{{
    """
    Here's the idea: pass in the database and the trainers/validators list (that is,
    the list of indices of the training and validation set within the grand set)
    then generate any additional data if needed
    """
    with open(ds.inpf) as f:
        inp = json.load(f)
    ref = copy.copy(inp['setup']['ref'])

    # pull grand representations and SCF/pred data for validation set
    # {{{
    # NOTE: the grand data set generally contains the validation set
    # so the positions of trainers (at least by the k-means algorithm)
    # is actually determined only for the subset of data which does not
    # contain the validators
    if 'remove' in kwargs:
        noval_reps = np.delete(ds.grand["representations"],validators,axis=0)
        t_reps = [noval_reps[i] for i in trainers]
    else:
        t_reps = [ds.grand['representations'][i] for i in trainers]
    p_CORR_E = [ds.grand['values'][i] for i in validators]
    v_reps = [ds.grand['representations'][i] for i in validators]
    v_SCF_list = [ds.grand['reference'][i] for i in validators]
    # }}}

    if ds.valtype == ds.predtype: # all data already computed
    # {{{
        print("Sorting training/validation sets from grand training data.")
        if 'remove' in kwargs:
            noval_E = np.delete(ds.grand["values"],validators,axis=0)
            t_CORR_E = [noval_E[i] for i in trainers]
        else:
            t_CORR_E = [ds.grand["values"][i] for i in trainers]
        v_CORR_E = [ds.grand["values"][i] for i in validators]
        v_SCF_list = [ds.grand["reference"][i] for i in validators]
    # }}}

    else: # we need valtype values for training set and (possibly) validation set
    # {{{
        # NOTE: while we need validation set reps for prediction, the energies
        # are used solely for the purpose of graphing the "true" energies against
        # the ML-E. Thus in a "real" scenario, additional calculations on the
        # validation set will not be done.
        while inp['data']['train_generated']:
            print("Loading training data from file.")
            try:
                t_CORR_E = np.load('train_corr_list.npy').tolist()
                break
            except FileNotFoundError:
                print("Data not found. Proceeding to data generation.")
                inp['data']['train_generated'] = False
        else:
            print("Generating training set data . . .")
            bot = float(inp['setup']['grange'][0])
            top = float(inp['setup']['grange'][1])
            pes = np.linspace(bot,top,ds.N)
            if 'remove' in kwargs:
                pes = np.delete(pes,validators,axis=0)
            pts = [pes[i] for i in trainers]
            t_CORR_E = datahelper.data_gen(ds,pts,ds.valtype)
            np.save('train_corr_list.npy',t_CORR_E)
            inp['data']['train_generated'] = True
        # update the input
        with open(ds.inpf,'w') as f:
            json.dump(inp, f, indent=4)

        if ref == True:
            while inp['data']['valid_generated']:
                print("Load validation data from file.")
                try:
                    v_CORR_E = np.load('valid_corr_list.npy').tolist()
                    break
                except FileNotFoundError:
                    print("Data not found. Proceeding to data generation.")
                    inp['data']['grand_generated'] = False
            else:
                print("Generating validation set data . . .")
                bot = float(inp['setup']['grange'][0])
                top = float(inp['setup']['grange'][1])
                pes = np.linspace(bot,top,ds.N)
                pts = [pes[i] for i in validators]
                v_CORR_E = []
                if ref == True:
                    v_CORR_E = datahelper.data_gen(ds,pts,ds.valtype)
                if ref == True:
                    np.save('valid_corr_list.npy',v_CORR_E)
                inp['data']['valid_generated'] = True
    
            # update the input
            with open(ds.inpf,'w') as f:
                json.dump(inp, f, indent=4)
    # }}}

    # internally shift all corr E by the avg training set corr E
    t_CORR_avg = np.mean(t_CORR_E)
    t_CORR_E = np.subtract(t_CORR_E,t_CORR_avg)

    # model determination (`s` and `l` hypers, then `a` coefficients)
    # {{{
    # train the hypers
    if inp['data']['hypers']:
        print("Loading hyperparameters from {}".format(ds.inpf))
        s = inp['data']['s']
        l = inp['data']['l']
    else:
        s, l = find_hypers(ds,t_CORR_E,t_reps)
        inp['data']['hypers'] = True
        inp['data']['s'] = s
        inp['data']['l'] = l
        with open(ds.inpf,'w') as f:
            json.dump(inp, f, indent=4)

    # train for alpha
    if inp['data']['a']:
        print("Loading coefficients from {}".format(ds.inpf)) 
        alpha = np.asarray(inp['data']['a'])
    else:
        print("Model training using s = {} and l = {} . . .".format(s,l))
        alpha = train(t_reps,t_CORR_E,s,l)
        inp['data']['a'] = alpha.tolist()
        with open(ds.inpf,'w') as f:
            json.dump(inp, f, indent=4)
    # }}}

    # predict E across the validators
    print("Predicting validation set . . .")
    pred_E_list = pred(v_reps,t_reps,alpha,s,l)

    pred_E_list = np.add(pred_E_list,t_CORR_avg)
    pred_E_list = np.add(pred_E_list,v_SCF_list)

    p_E_list = np.add(p_CORR_E,v_SCF_list)
    
    return_dict = {"Predictions": pred_E_list,
                   "SCF": v_SCF_list,
                   "Predtype": p_E_list,
                   "Average": t_CORR_avg}
    if ref == True:
        v_E_list = np.add(v_CORR_E,v_SCF_list)
        return_dict["Validations"] = v_E_list

    return return_dict
# }}}

def pred(x,xt,a,s=0.05,l=0):
# {{{
    '''
    predict answers "Y" across entire PES
    given x (full set of reps), xt (training reps), and a (alpha)
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

def train(x,y,s=0.05,l=0):
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

def find_hypers(ds,t_y,t_reps,**kwargs):
# {{{
    # TODO: it would be nice to be able to take an optional argument which
    # determines the hyperparameters using different methods
    if "k" in kwargs: 
        k = kwargs['k']
    else:
        k = ds.M
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
            a = train(tr_x,tr_y,s,l)
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
        print("s = {}\nl = {}\nmse = {}\n\n".format(s,l,mse))
    s = np.mean(s_list)
    l = np.mean(l_list)
    return s,l 
# }}}
