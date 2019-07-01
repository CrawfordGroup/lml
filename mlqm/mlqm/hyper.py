import numpy as np
import copy


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

