import numpy as np
from numpy import linalg as la
import psi4
psi4.core.be_quiet()
import mesp
import matplotlib
import matplotlib.pyplot as plt
import random
import copy

def do_krr(M=12,N=50,st=0.05,s=5,l=0):# {{{
    '''
    kernel ridge regression with M training points and N total representations
    s used as the stdv for gaussian kernel generation
    l used for ridge regression
    need another optional s_t to push into the TATR generation...
    '''
    bas = "def2-TZVP"
#    bas = "dz"
    pes = np.linspace(0.5,2,N) # N evenly-spaced representations on the PES

#    ### TEST N2 TATR #### {{{
#    geom = """
#        N
#        N 1 1.1
#        symmetry c1
#        """
#    mol = mesp.Molecule('N2',geom,bas)
#    tatr = make_tatr(mol,graph=True)# }}}

    # generate the total data set # {{{
    tatr_list = [] # hold TATRs
    E_list = [] # hold E
    for i in range(0,N):
        geom = """
            H
            H 1 """ + str(pes[i]) + """
            symmetry c1
        """
        mol = mesp.Molecule('H2',geom,bas)
        tatr = make_tatr(mol,150,st)
        tatr_list.append(tatr)
        E_list.append(mol.E_CCSD)# }}}

    # generate the training set and save their positions# {{{
    trainers = gen_train(tatr_list, M, graph=False) 
    t_pos = [pes[i] for i in trainers]

    t_tatr = [] # training TATRs
    t_E = [] # training energies
    for i in trainers:
        t_tatr.append(tatr_list[i])
        t_E.append(E_list[i])# }}}

    # k-fold cross-validation to tune s
    k = M
    step = 100
    max_it = 500
#    grad_conv = 1E-16 
    grad_conv = 1E-6
    s_new = s+1
    s_list = [s,s_new]

    l_new = l+(l*0.5)
    l_list = [l,l_new]

    cv_error = []
    cv_error_old, pred_E_list, mse_list = cross_validate(t_tatr,t_E,k,s,l)
    cv_error = np.append(cv_error,cv_error_old)

#    cv_error = [[],[]]
#    cv_error_s_old, pred_E_list, mse_list = cross_validate(t_tatr,t_E,k,s,l)
#    cv_error_l_old, pred_E_list, mse_list = cross_validate(t_tatr,t_E,k,s,l)
#    tmp1 = np.append(cv_error[0],[cv_error_s_old],axis=0)
#    tmp2 = np.append(cv_error[1],[cv_error_l_old],axis=0)
#    cv_error = np.concatenate(([tmp1],[tmp2]))
#    print("first cv_error: {}".format(cv_error))
    for it in range(1,max_it+1):
        cv_error_new, pred_E_list, mse_list = cross_validate(t_tatr,t_E,k,s_new,l)
        cv_error = np.append(cv_error,cv_error_new)

#        cv_error_s_new, pred_E_list, mse_list = cross_validate(t_tatr,t_E,k,s_new,l)
#        cv_error_l_new, pred_E_list, mse_list = cross_validate(t_tatr,t_E,k,s,l_new)
#        tmp1 = np.append(cv_error[0],[cv_error_s_new],axis=0)
#        tmp2 = np.append(cv_error[1],[cv_error_l_new],axis=0)
#        cv_error = np.concatenate(([tmp1],[tmp2]))
#        print("next cv_error: {}".format(cv_error))

        grad = np.gradient(cv_error,s_list)
        abs_grad = [abs(i) for i in grad]

#        grad = np.gradient(cv_error,s_list,l_list)
#        print(grad)
#        abs_grad = np.asarray([abs(i) for i in grad])
#        abs_grad = abs_grad.ravel()
#        print(min(abs_grad))

        if min(abs_grad) < grad_conv:
            print("converged in {} iterations".format(it))
            print("s value: {}".format(s_new))
#            print("l value: {}".format(l_new))
            print("l value: {}".format(l))
            #print("CV error: {}".format(cv_error_new))
            break
        elif it == max_it:
            print("too many iterations")
            print("s value: {}".format(s_new))
            print("l value: {}".format(l))
#            print("l value: {}".format(l_new))
            #print("CV error: {}".format(cv_error_new))
            break
        else:
            s = s_new
            s_new = s - step*grad[-1]
#            s_new = s - step*grad[0][-1]
            s_list.append(s_new)
#            l = l_new
#            l_new = l - step*grad[0][-1]
#            l_list.append(l_new)

    # train for alpha
    alpha = train(t_tatr,t_E,s_new,l)

    # predict E across the PES
    pred_E_list = pred(tatr_list,t_tatr,alpha,s_new,l)

    # plot
    E_list = np.asarray(E_list)
    pred_E_list = np.asarray(pred_E_list)

    plt.figure(1)
    plt.plot(pes,E_list,'b-o',label='PES')
    plt.plot(pes,pred_E_list,'r^',ms=2,label='ML-{}'.format(M))
    plt.plot(t_pos,[-1.18 for i in range(0,len(t_pos))],'go',label='Training points')
    plt.axis([0.25,2.0,-2.5,-0.8])
    plt.xlabel('r/Angstrom')
    plt.ylabel('Energy/E_h')
    plt.legend()
    plt.show()
    # }}}

def gaus(x, u, s):# {{{
    '''
    return a gaussian centered on u with width s
    note: we're using x within [-1,1]
    '''
    return np.exp(-(x-u)**2 / (2.0*s**2))# }}}

def make_tatr(mol,x=150,st=0.05,graph=False):# {{{
    '''
    make t-amp tensor representation
    pass in a mesp molecule and (optional) the number of points
    and (optional) the sigma for the gaussian
    '''
    mesp.do_ccsd(mol, e_conv=1e-8, save_t=True)

    # if there aren't enough t's, just keep them all
    # probably redundant...
#    if len(mol.t1) < x1:
#        x1 = len(mol.t1.ravel())
#    if len(mol.t2) < x2:
#        x2 = len(mol.t2.ravel())

    t1 = np.sort(mol.t1.ravel())[-x:]
    t2 = np.sort(mol.t2.ravel())[-x:]
#    t = np.concatenate((t1,t2),axis=None)
#    print("Total t1 amplitudes: {}".format(len(t1)))
#    print("Total t2 amplitudes: {}".format(len(t2)))
#    print("Total t amplitudes: {}".format(len(t)))
    
    tatr = [] # store eq vals
    tatr1 = [] # singles tatrs
    tatr2 = [] # doubles tatrs
#    x_list = np.linspace(-1,1,(x1+x2)) # if I get rid of the "redundant" code above, then I need to use the lengths of the t-vector
    x_list = np.linspace(-1,1,x)
#    print("x_list len: {}".format(len(x_list)))
    for i in range(0,x):
        val1 = 0
        val2 = 0
        for t_1 in range(0,len(t1)):
            val1 += gaus(x_list[i],t1[t_1],st)
        for t_2 in range(0,len(t2)):
            val2 += gaus(x_list[i],t2[t_2],st)
        tatr1.append(val1)
        tatr2.append(val2)
#        tatr.append(np.concatenate((tatr1,tatr2),axis=None))
#        tatr.append(np.concatenate((tatr1[i],tatr2[i])))
        tatr.append(val1)
        tatr.append(val2)
#        print("tatr at x = {}: {}\n".format(x_list[i],tatr[i]))
#    print("tatr at x = {}: {}\n".format(x_list[149],tatr[149]))

#    for N in range(0,len(t2)):
#        tatr.append(gaus(x_list[N], t2[N], s))

    if graph:
#        print(tatr)
        plt.figure(1)
        plt.plot(x_list,tatr1,label="singles")
        plt.plot(x_list,tatr2,label="doubles")
        plt.legend()
#        plt.yscale("log")
        plt.show()
    return np.asarray(tatr)# }}}

def gen_train(tatr_list,M,cen_conv=1e-12,max_iter=20,graph=False):# {{{
    '''
    Generate a training set by the k-means algorithm
    Try to converge the centers before max_iter
    '''
    print("Generating training set . . .")
#    print("Total data set: {}".format(tatr_list))
    trainers = random.sample(range(0,len(tatr_list)),M)
#    print("Initial trainer indices: {}".format(trainers))

    # establish initial centers of clusters
    centers = []
    for i in range(0,M):
        centers.append(tatr_list[trainers[i]])
    centers = np.asarray(centers)
    cen_diffs = [cen_conv + 1]
#    print("Initial centers: {}".format(centers))

    # iterate re-centering
    for _ in range(0,max_iter):
        # determine mapping of TATRs to clusters 
        t_map = [[] for __ in range(M)] # see map below
        # [[cluster1: (tatr,#), (tatr,#), . . .],
        #  [cluster2: (tatr,#), (tatr,#), . . .]]
        diffs = [[] for __ in range(M)] # see map below
        # [[cluster1: tatr1 dist, tatr2 dist, . . .], 
        #  [cluster2: tatr1 dist, tatr2 dist, . . .]]
        for i in range(0,len(tatr_list)): # loop over TATRs
            for j in range(0,len(centers)): # loop over centers
#                print("tatr_list[i]: {}\n\ncenters[j]: {}".format(tatr_list[i],centers[j]))
#                print("len of tatr_list[i]: {}\nlen of centers[j]: {}".format(len(tatr_list[i]),len(centers[j])))
                diff = la.norm(tatr_list[i] - centers[j]) # norm of diff btwn TATR & center
                diffs[j].append(diff)
                if j == 0: # first is always "closest" to start
                    cnt = 0
                    old_diff = diff
                elif abs(diff) < old_diff: # then see if others are closer
                    cnt = j # if so, store the TATR map
                    old_diff = diff # and keep track of the smallest diff
            t_map[cnt].append((tatr_list[i],i))
        if graph:
#            print("t_map for iter {}: {}".format(_,t_map))
#            print("diffs for iter {}: {}".format(_,diffs))
            for c in range(0,M):
                print("Size of cluster {} in iter {}: {}".format(c,_,len(t_map[c])))
    
        if (_ >= 1) and (max(cen_diffs) <= cen_conv): # force one iteration
            print("Converged training set in {} iterations!".format(_+1))
#            print("Diffs: {}".format(diffs))
            trainers = []
            for train in range(0,M):
                trainers.append(np.argmin(diffs[train]))
#                print("Smallest diff in cluster {} is TATR {}".format(train,np.argmin(diffs[train])))
            return trainers

        # determine new centers and differences
        cen_diffs = []
        for cen in range(0,M):
            new_cen = np.mean([x[0] for x in t_map[cen]],axis=0) # list comprehension of tuples
#            print("Old axis: {}".format([x[0] for x in t_map[cen]]))
            cen_diffs.append(np.mean(new_cen - centers[cen]))
            centers[cen] = new_cen
#        print("New centers for iter {}: {}".format(_,centers))# }}}

def make_k(tatr1, tatr2, s):# {{{
    '''
    return a gaussian kernel from two TATRs
    '''
    tmp = la.norm(tatr1 - tatr2)
    return np.exp(-1 * tmp**2 / (2.0*s**2))# }}}

def solve_alpha(K,Y,l=0):# {{{
    '''
    solve the kernel ridge regression expression for alpha
    give Kernel matrix and training matrix
    optional lambda parameter
    '''
    return la.solve((K-l*np.eye(len(Y))),Y)# }}}

def train(x,y,s=0.05,l=0):# {{{
    '''
    optimize parameter(s) "a" by solving linear equations
    given x (training TATRs) and y (training energies)
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

    return a# }}}

def pred(x,xt,a,s=0.05,l=0):# {{{
    '''
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
    
    return Y# }}}

def loss(y,y_p):# {{{
    '''
    Calculate L2 (squared) loss, return mean squared error
    given true and predicted values
    '''
    l = [(y[i] - y_p[i])**2 for i in range(len(y))]
    return (sum(l)/len(y))# }}}

def cross_validate(x_data,y_data,k,s=0.05,l=0):# {{{
    '''
    k-fold cross validation: k folds, one is kept for validation
    "rotate" folds, re-validate, repeat to finish
    given x and y values to regress, number of folds k
    s and l hyperparameters (kernel width and regularization)
    return average error and predicted values + their errors for each fold 
    may also want to return the parameter (a)...
    '''
    # split full data into k folds
    y_data = np.array_split(y_data,k) 
    x_data = np.array_split(x_data,k) 

    # pop out validation set, train/validate, repeat
    y_p_list = [] # predicted values
    mse_list = [] # mean squared errors
    for val in range(0,k): # loop over validation sets
        tr_y = copy.deepcopy(y_data) # copy full data set
        tr_x = copy.deepcopy(x_data)
        val_y = tr_y.pop(val) # pop out the validation set
        val_x = tr_x.pop(val)
        tr_y = np.concatenate(tr_y) # re-form training set
        tr_x = np.concatenate(tr_x) 

        a = train(tr_x,tr_y,s,l) # train using training set
        y_p = pred(val_x,tr_x,a,s,l) # predict validation set
        mse = loss(val_y,y_p) # calculate loss
        y_p_list.append(y_p) # save the answers
        mse_list.append(mse) # save the error

    return np.mean(mse_list), y_p_list, mse_list# }}}

if __name__ == "__main__":
    do_krr()
