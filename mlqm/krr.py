import numpy as np
from numpy import linalg as la
import psi4
psi4.core.be_quiet()
import mesp
import matplotlib
import matplotlib.pyplot as plt
import random
import copy

def do_krr(M=12,N=200,st=0.05,s=10000,l=0,K=30,save=True):# {{{
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
    try:
        print("Loading data set . . .")
        tatr_list = np.load('t_list.npy').tolist()
        E_CCSD_CORR_list = np.load('corr_list.npy').tolist()
        E_SCF_list = np.load('scf_list.npy').tolist()
    except FileNotFoundError:
        print("Generating data set . . .")
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
            E_SCF_list.append(mol.E_SCF)
        if save:
            np.save('t_list.npy',tatr_list)
            np.save('corr_list.npy',E_CCSD_CORR_list)
            np.save('scf_list.npy',E_SCF_list)# }}}

    # generate the training/validation sets# {{{
    print("Generating training set . . .")
    trainers = gen_train(tatr_list, M, K) 

#    print("Using test training set.")
#    trainers = sorted([152, 139, 35, 138, 112, 32, 106, 172, 57, 143, 52, 182], reverse=True) 
    # s=10000,l=0 is good

#    trainers = sorted([98, 116, 178, 47, 8, 138, 182, 32, 157, 75, 144, 143], reverse=True) 
    # s=100,l=0 is good

#    trainers = [182, 178, 174, 171, 144, 136, 125, 111, 109, 57, 47, 32]
#    trainers = [184, 180, 176, 172, 144, 136, 124, 112, 108, 56, 48, 32]
    # the above is completely composed of validation points- should get exact fit at these points

    print("Training set: {}".format(trainers))
    t_pos = [pes[i] for i in trainers]

    vals = sorted([i for i in range(0,200,4)], reverse=True) # validation set

    t_tatr = [] # training TATRs
    t_CORR_E = [] # training energies
    v_tatr = [] # validation TATRs
    v_CORR_E = [] # validation energies
    v_SCF = [] # validation SCF energies
    v_PES = [] # validation PES

    for i in trainers:
        t_tatr.append(tatr_list[i]) # assuming I don't have to pop the trainers out of set
        t_CORR_E.append(E_CCSD_CORR_list[i])
    for i in vals:
#        vtatr = tatr_list.pop(i) # make _sure_ to pop vals out of set
        vtatr = tatr_list[i]
        vE = E_CCSD_CORR_list.pop(i)
        v_tatr.append(vtatr)
        v_CORR_E.append(vE)
        v_SCF.append(E_SCF_list[i])
        v_PES.append(pes[i])

    # internally shift all corr E by the avg training set corr E
    t_CORR_avg = np.mean(t_CORR_E)
    t_CORR_E = np.subtract(t_CORR_E,t_CORR_avg)
    v_CORR_E = np.subtract(v_CORR_E,t_CORR_avg)
        # }}}

    s_new, l = log_space_cv(t_CORR_E,t_tatr,s,l,k=M)
#     USE BELOW FOR TESTING
#    print("WARNING: Using static s and l values!")
#    l = 0
#    s_new = s 
#    s_new = 5737.081137
#    s_new = 11657780
#    l = 7e-5
#    l = 7e-26

    # train for alpha
    print("Model training using s = {} and l = {} . . .".format(s_new,l))
    alpha = train(t_tatr,t_CORR_E,s_new,l)

    # predict E across the PES
    print("Predicting PES . . .")
    pred_E_list = pred(v_tatr,t_tatr,alpha,s_new,l)

    # plot
    v_E_list = np.add(v_CORR_E,t_CORR_avg)
    v_E_list = np.add(v_E_list,v_SCF)
    pred_E_list = np.add(pred_E_list,t_CORR_avg)
    pred_E_list = np.add(pred_E_list,v_SCF)

    plt.figure(1)
    plt.plot(v_PES,v_E_list,'b-o',label='CCSD PES',linewidth=3)
    plt.plot(v_PES,v_SCF,'y-',label='SCF')
    plt.plot(v_PES,pred_E_list,'r-^',ms=2,label='CCSD/ML-{}'.format(M),linewidth=2)
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
    Reverse the ordering so pop() can be done in succession on grand training set
    '''
    trainers = []
    t_map, close_pts = k_means_loop(tatr_list,M,K)
    for train in range(0,M): # loop over centers, grab position of training points from each
        trainers.append(t_map[train][close_pts[train]][1])
    return sorted(trainers, reverse=True)# }}}

def make_k(tatr1, tatr2, s):# {{{
    '''
    return a gaussian kernel from two TATRs
    '''
    tatr1 = np.asarray(tatr1)
    tatr2 = np.asarray(tatr2)
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
#    lam = l*np.eye(N)[:M,:]
#    lam = lam[:len(k[0])]
    for i in range(0,N):
#        print("a is {} long\nk[{}] is {} long".format(len(a),i,len(k[i])))
        Y.append(np.dot(a,k[i]))
#        print("k[{}] = {}".format(i,k[i]))
#        print("lam = {}".format(lam))
#        print("lam is {}x{}".format(len(lam),len(lam)))
#        print("lam[{}] is size {}".format(i,lam[i].size))
#        Y.append(np.dot(a,np.subtract(k[i],lam[:,i])))
#    print("Y = {}".format(Y))
    
    return Y# }}}

def loss(y,y_p):# {{{
    '''
    Calculate L2 (squared) loss, return mean squared error
    given true and predicted values
    '''
    l = [(y[i] - y_p[i])**2 for i in range(len(y))]
    return (sum(l)/len(y))# }}}

def grid_search(tr_x,val_x,tr_y,val_y):# {{{
    mse = 1E9
    s_list = np.logspace(-5,8,num=16)
    l_list = np.logspace(-16,16,num=40,base=-10)
#    l_list = np.zeros(50)
#    print("WARNING: Setting all lambda values to zero!")
    for s in s_list:
        for l in l_list:
            a = train(tr_x,tr_y,s,l)
            y_p = pred(val_x,tr_x,a,s,l)
            new_mse = abs(loss(val_y,y_p))
            if new_mse <= mse:
                s_f = s
                l_f = l
                mse = new_mse
    return s_f, l_f, mse# }}}

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

def log_space_cv(y_data,x_data,s,l,k=5,step=100,max_it=3000,grad_conv=1E-8):# {{{
    '''
    Test function- let's do cross-validation by searching over a log space for
    my hyperparameters. Optimize a hyperparameter(s) for one fold, push that into the
    next, optimize, and repeat. 
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
#    den = sum(1/mse for mse in mse_list)
#    s = sum(s_list[i] / mse_list[i] for i in range(len(s_list))) / den
#    l = sum(l_list[i] / mse_list[i] for i in range(len(l_list))) / den
    s = np.mean(s_list)
    l = np.mean(l_list)
    return s,l # }}}

# old cross-validation w/ gradient descent# {{{
#        s_new = s+1
#        fold_error = [] # list of errors for this fold
#        fold_s = [s,s_new]
#
#        for it in range(1,max_it+1):
##            print("it {}".format(it))
#            if it == 1:
#                a = train(tr_x,tr_y,s,l) # train using training set
#                y_p = pred(val_x,tr_x,a,s,l) # predict validation set
#                mse = loss(val_y,y_p) # calculate loss
#                fold_error = np.append(fold_error,mse)
#            else:
#                a = train(tr_x,tr_y,s_new,l) # train using training set
#                y_p = pred(val_x,tr_x,a,s_new,l) # predict validation set
#                new_mse = loss(val_y,y_p) # calculate loss
#                fold_error = np.append(fold_error,new_mse) # store error for this fold
#    
#                grad = np.gradient(fold_error,fold_s)
#                abs_grad = [abs(i) for i in grad]
#        
#                if min(abs_grad) < grad_conv:
#                    s = s_new
#                    print("converged in {} iterations".format(it))
#                    print("s value: {}".format(s_new))
#                    print("l value: {}".format(l)) # s-only learning
#                    #print("CV error: {}".format(cv_error_new))
#                    break
#                elif it == max_it:
#                    s = s_new
#                    print("too many iterations")
#                    print("s value: {}".format(s_new))
#                    print("l value: {}".format(l)) # s-only learning
#                    break
#                else:
#                    s = s_new
#                    s_new = s - step*grad[-1] # s-only learning
#                    fold_s.append(s_new)
#    return s# }}}

if __name__ == "__main__":
    do_krr()
