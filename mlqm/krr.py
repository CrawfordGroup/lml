import numpy as np
from numpy import linalg as la
import psi4
psi4.core.be_quiet()
import mesp
import matplotlib
import matplotlib.pyplot as plt
import random

def do_krr(M=12,N=200,s=0.2,la=0):# {{{
    '''
    kernel ridge regression with M training points and N total representations
    s used as the stdv for gaussian kernel generation
    s=0.05 will learn nothing, but match training points exactly (k too small)
    la used for ridge regression
    need another optional s_t to push into the TATR generation...
    '''
    bas = "def2-TZVP"
#    bas = "dz"
    l = np.linspace(0.5,2,N) # N evenly-spaced representations on the PES

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
            H 1 """ + str(l[i]) + """
            symmetry c1
        """
        mol = mesp.Molecule('H2',geom,bas)
        tatr = make_tatr(mol)
        tatr_list.append(tatr)
        E_list.append(mol.E_CCSD)# }}}

    # generate the training set and save their positions
    trainers = gen_train(tatr_list, M, graph=False) 
    t_pos = [l[i] for i in trainers]

    # make the K-trainer matrix
    K = np.zeros((M,M))
    for i in range(0,M):
        for j in range(0,M):
            K[i][j] = make_k(tatr_list[i],tatr_list[j],s)
    
    # grab the real answers and solve for alpha
    Y = []
    for i in range(0,M):
        Y.append(E_list[trainers[i]])
    alpha = solve_alpha(K,Y)

    # make the full k matrix [k(train,predict)]
    k = []
    for i in range(0,N):
        k.append([])
        for j in range(0,M):
            k[i].append(make_k(tatr_list[trainers[j]],tatr_list[i],s))

    # predict energy for the entire PES
    pred_E_list = [] # 
    for i in range(0,N):
        pred_E_list.append(np.dot(alpha,k[i]))

    E_list = np.asarray(E_list)
    pred_E_list = np.asarray(pred_E_list)

    # plot# {{{
    plt.figure(1)
    plt.plot(l,E_list,'b-o',label='PES')
    plt.plot(l,pred_E_list,'r-',label='ML-{}'.format(M))
    plt.plot(t_pos,[-1.18 for i in range(0,len(t_pos))],'go',label='Training points')
    plt.axis([0.25,2.0,-2.5,-0.8])
    plt.xlabel('r/Angstrom')
    plt.ylabel('Energy/E_h')
    plt.legend()
    plt.show()# }}}
    # }}}

def gaus(x, u, s):# {{{
    '''
    return a gaussian centered on u with width s
    note: we're using x within [-1,1]
    '''
    return np.exp(-(x-u)**2 / (2.0*s**2))# }}}

def make_tatr(mol,x=150,s=0.05,graph=False):# {{{
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
            val1 += gaus(x_list[i],t1[t_1],s)
        for t_2 in range(0,len(t2)):
            val2 += gaus(x_list[i],t2[t_2],s)
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
    print("norm = {}".format(tmp))
    return np.exp(-1 * tmp**2 / (2.0*s**2))# }}}

def solve_alpha(K,Y,l=0):# {{{
    '''
    solve the kernel ridge regression expression for alpha
    give Kernel matrix and training matrix
    optional lambda parameter
    '''
    return la.solve((K-l*np.eye(len(Y))),Y)# }}}

def cross_validate(f_data,k):
    '''
    k-fold cross validation: k folds, one is kept for validation
    "rotate" folds, re-validate, repeat to finish
    return average error
    '''
    # split full data into k folds
    f_data = np.split(f_data,k) 

    

    return 0

if __name__ == "__main__":
    do_krr()
