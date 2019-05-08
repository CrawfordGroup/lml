import numpy as np
from numpy import linalg as la
import psi4
psi4.core.be_quiet()
import mesp
import matplotlib
import matplotlib.pyplot as plt
import random
import copy

def do_krr(M=12,N=50,st=0.05,s=5,l=0.0005,K=30):# {{{
    '''
    kernel ridge regression with M training points and N total representations
    st used for tatr generation
    s used as the stdv for gaussian kernel generation
    l used for ridge regression
    K repeated k-means algorithms, best clustering chosen
    '''
    bas = "def2-TZVP"
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
    E_CCSD_CORR_list = [] # hold CCSD correlation energy
    E_SCF_list = [] # hold SCF energy
    for i in range(0,N):
        geom = """
            H
            H 1 """ + str(pes[i]) + """
            symmetry c1
        """
        mol = mesp.Molecule('H2',geom,bas)
        tatr = make_tatr(mol,150,st)
        tatr_list.append(tatr)
        E_CCSD_CORR_list.append(mol.E_CCSD_CORR)
        E_SCF_list.append(mol.E_SCF)# }}}

    E_CCSD_CORR_avg = np.mean(E_CCSD_CORR_list)
    E_CCSD_CORR_list = np.subtract(E_CCSD_CORR_list,E_CCSD_CORR_avg)

    # generate the training set and save their positions# {{{
    print("Generating training set . . .")
    trainers = gen_train(tatr_list, M, K) 
    t_pos = [pes[i] for i in trainers]

    t_tatr = [] # training TATRs
    t_E = [] # training energies
    for i in trainers:
        t_tatr.append(tatr_list[i])
        t_E.append(E_CCSD_CORR_list[i])# }}}

    # k-fold cross-validation to tune s
    print("Cross-validating s . . .")
    k = 5
    step = 100
    max_it = 1000
    grad_conv = 1E-5 
    s_new = s+1
    s_list = [s,s_new]

    l_new = l+(l*0.5)
    l_list = [l,l_new]

    cv_error = []
    cv_error_old, pred_E_list, mse_list = cross_validate(t_tatr,t_E,k,s,l)
    cv_error = np.append(cv_error,cv_error_old)

    for it in range(1,max_it+1):
        cv_error_new, pred_E_list, mse_list = cross_validate(t_tatr,t_E,k,s_new,l)
        cv_error = np.append(cv_error,cv_error_new)

        grad = np.gradient(cv_error,s_list)
        abs_grad = [abs(i) for i in grad]

        if min(abs_grad) < grad_conv:
            print("converged in {} iterations".format(it))
            print("s value: {}".format(s_new))
            print("l value: {}".format(l)) # s-only learning
            #print("CV error: {}".format(cv_error_new))
            break
        elif it == max_it:
            print("too many iterations")
            print("s value: {}".format(s_new))
            print("l value: {}".format(l)) # s-only learning
            break
        else:
            s = s_new
            s_new = s - step*grad[-1] # s-only learning
            s_list.append(s_new)

    print("final error (before prediction): {}".format(cv_error_new))

    # train for alpha
    alpha = train(t_tatr,t_E,s_new,l)

    # predict E across the PES
    pred_E_list = pred(tatr_list,t_tatr,alpha,s_new,l)

    # plot
    E_CCSD_CORR_list = np.asarray(E_CCSD_CORR_list)
    E_CCSD_CORR_list = np.add(E_CCSD_CORR_list,E_CCSD_CORR_avg)
    E_SCF_list = np.asarray(E_SCF_list)
    E_list = np.add(E_CCSD_CORR_list,E_SCF_list)
    pred_E_list = np.asarray(pred_E_list)
    pred_E_list = np.add(pred_E_list,E_CCSD_CORR_avg)
    pred_E_list = np.add(pred_E_list,E_SCF_list)

    plt.figure(1)
    plt.plot(pes,E_list,'b-o',label='CCSD PES')
    plt.plot(pes,E_SCF_list,'y-',label='SCF')
    plt.plot(pes,pred_E_list,'r^',ms=2,label='CCSD/ML-{}'.format(M))
    plt.plot(t_pos,[-1.18 for i in range(0,len(t_pos))],'go',label='Training points')
    plt.axis([0.25,2.0,-2.5,-0.8])
    plt.xlabel('r/Angstrom')
    plt.ylabel('Energy/$E_h$')
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

def cluster(pts,M,cens):# {{{
    '''
    Assign each point in pts to one of M clusters centered on cens
    Return the cluster-point maps and the point-center differences
    '''
    diffs = [[] for _ in range(M)] # [[cluster 1: pt1 diff, pt2 diff, . . .],[cluster 2: pt1 diff, . . .]]
    maps = [[] for _ in range(M)] # [[cluster 1: (pt,#), . . .],[cluster 2: (pt,#), . . .]]
    for i in range(0,len(pts)): # loop over points
        for j in range(0,len(cens)): # loop over centers
            diff = la.norm(pts[i] - cens[j]) # norm of diff btwn point and center
            if j==0: # first center is always "closest" to pt
                cnt = 0
                old_diff = diff
            elif abs(diff) < old_diff: # see if other centers are closer
                cnt = j # if so, store the pt map
                old_diff = diff # and keep track of the smallest diff
        maps[cnt].append((pts[i],i)) # store the actual point, and its "number" in the list
    return maps# }}}

def recenter(M,maps):# {{{
    '''
    Return centers (vector means) of each of the M cluster in maps
    '''
    new_cens = []
    for cen in range(0,M):
        cluster = [x[0] for x in maps[cen]]
        new_cen = np.mean([x[0] for x in maps[cen]],axis=0)
        new_cens.append(new_cen)
    return np.asarray(new_cens)# }}}

def k_means(pts,M):# {{{
    '''
    List of initial data points pts
    Number of centers M
    '''
    points = random.sample(range(0,len(pts)),M)
    cens = np.asarray([pts[points[i]] for i in range(0,M)])
    maps = cluster(pts,M,cens)
    diffs = [1]
    while max(diffs) > 0:
        new_cens = recenter(M,maps)
        maps = cluster(pts,M,new_cens)
                                                                            
        diffs = [np.mean(cens[i] - new_cens[i]) for i in range(0,len(new_cens))]
        diffs = np.absolute(diffs)
        cens = new_cens
    return np.asarray(maps)# }}}

def cen_err(maps):# {{{
    '''
    Take in the center->point map
    Return the stddv of the mean center->point diffs
    Also return the positions of the closest points to each center
    '''
    errors = []
    close_pts = []
    for i in range(0,len(maps)): # loop over centers
        cen = np.mean([c[0] for c in maps[i]],axis=0) # mean of coords in center i
        diffs = [] # hold diffs for center i
        for j in range(0,len(maps[i])): # loop over points
            diffs.append(la.norm(maps[i][j][0] - cen)) # norm of diff btwn point and cen
        errors.append(np.mean(diffs)) # store the mean error for center i
        close_pts.append(np.argmin(diffs)) # store the position of the closest point to center i
    return np.std(errors), close_pts# }}}

def k_means_loop(pts,M,K):# {{{
    '''
    Loop the k-means algorithm K times
    Return the best clustered map and position of closest points to centers
    '''
    maps_list = [] # hold all the maps
    errors = [] # hold all the errors
    close_pts_list = [] # hold all the closest points
    for i in range(0,K):
        maps = k_means(pts,M)
        maps_list.append(maps)
        error, close_pts = cen_err(maps)
        errors.append(error)
        close_pts_list.append(close_pts)
    best = np.argmin(errors) # lowest stdv of mean point-center diff is best
    return maps_list[best], close_pts_list[best]# }}}

def gen_train(tatr_list,M,K):# {{{
    '''
    Generate a training set given the list of TATRs and the number of training points
    Converge the k-means algorithm K separate times, choose best clustering
    Data points closest to cluster centers are chosen as trainers
    '''
    trainers = []
    t_map, close_pts = k_means_loop(tatr_list,M,K)
    print("close pts: {}".format(close_pts))
    for train in range(0,M): # loop over centers, grab position of training points from each
        print("center {}".format(train))
        print("so it's pt {} in the t_map for center {}".format(close_pts[train],train))
        check = t_map[train]
        check2 = t_map[train][close_pts[train]]
        trainers.append(t_map[train][close_pts[train]][1])
        print("trainers: {}".format(trainers))
#    t_map, diffs = k_means_loop(tatr_list,M,K)
#    for train in range(0,M): # loop over the centers, find the closest point to each
#        trainers.append(np.argmin(diffs[train]))
    return trainers# }}}

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
