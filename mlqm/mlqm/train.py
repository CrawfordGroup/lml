import random
import numpy as np
from numpy import linalg as la
import datahelper

#
#def krr(treps,tvals,loss=datahelper.reg_l2,**kwargs):
#    # {{{
#    '''
#    inpf must contain AT LEAST:
#    inp['mol']['geom']: a geometry with a variable (gvar) given for displacement along PES
#    inp['setup']: 'M', 'N', 'st', 'K', 'gvar' (list), 'grange' (list), 'basis'
#    and empty/'false' spots for several to-be-filled fields.
#    See examples at github.com/bgpeyton/lml/mlqm/krr/examples/
#
#    Kernel Ridge Regression with M training points and N total representations
#    `valtype` is used for validation and high-level training data (IE, energies)
#    `predtype` is used for representations and predictions (IE, TATRs)
#    in short: "Predict `valtype`-level energies using `predtype`-level TATRs"
#    st used for tatr generation
#    K repeated k-means algorithms, best clustering chosen
#    save option to write TATRs and  energies to file
#
#    The algorithm will displace a diatomic over `grange[0]-grange[1]` Angstroms, generating
#    TATRs and corresponding energies at each point. Then, a training set of `M` TATRs 
#    will be selected by the k-means algorithm. Using this training set, the 
#    hyperparameters `s` and `l` (kernel width and regularization) will be determined by 
#    searching over a coarse logarithmic grid and minimizing the regularized L2 squared 
#    loss cost function. Once the hypers are thus determined, the regression coefficients 
#    `a` are trained by solving the corresponding linear equations. A validation set 
#    spread across the PES (not including any training points) is then predicted using the 
#    model, and the results are graphed.
#
#    Use from command line as:
#    >>>python krr.py inpf
#    '''
#
#    # generate the training/validation sets# {{{
#    if inp['data']['trainers']:
#        print("Loading training set from {} . . .".format(inpf))
#        trainers = inp['data']['trainers']
#        if len(trainers) != M:
#            print("Given training set does not have {} points! Re-generating . . .".format(M))
#            trainers = gen_train(tatr_list, M, K) 
#            inp['data']['trainers'] = trainers
#            with open(inpf,'w') as f:
#                json.dump(inp, f, indent=4)
#    else:
#        print("Generating training set . . .")
#        trainers = gen_train(tatr_list, M, K) 
#        inp['data']['trainers'] = trainers
#
#    # update the input
#    with open(inpf,'w') as f:
#        json.dump(inp, f, indent=4)
#
#    print("Training set: {}".format(trainers))
#    t_pos = [pes[i] for i in trainers] # training set positions for graphing purposes
#
#    vals = sorted([i for i in range(0,200,4)], reverse=True) # validation set
#    vals = sorted(list(set(vals) - (set(vals) & set(trainers))),reverse=True) # pull trainers out
#    print("Validation set: {}".format(vals))
#
#    t_tatr = [] # training TATRs
#    t_CORR_E = [] # training energies
#    v_tatr = [] # validation TATRs
#    v_CORR_E = [] # validation energies (for graphing with ref=True)
#    p_CORR_E = [] # validation energies with predtype (for graphing with ref=True)
#    v_SCF = [] # validation SCF energies
#    v_PES = [] # validation PES
#
#    if valtype == predtype:
#        print("Sorting training/validation sets from existing data.")
#        for i in trainers:
#            t_tatr.append(tatr_list[i]) # assuming I don't have to pop the trainers out of set
#            t_CORR_E.append(E_CORR_list[i])
#        for i in vals:
#            vtatr = tatr_list[i]
#            vE = E_CORR_list.pop(i)
#            v_tatr.append(vtatr)
#            v_CORR_E.append(vE)
#            v_SCF.append(E_SCF_list[i])
#            v_PES.append(pes[i])
#    else:
#        # for MP2->CCSD, we need MP2 TATRs (already computed) and CCSD energies
#        # for training set and validation set
#        # NOTE: while we need validation set TATRs for prediction, the energies
#        # are used solely for the purpose of graphing the "true" PES against
#        # the ML-PES. Thus in a "real" scenario, additional calculations on the
#        # validation set will not be done.
#        while inp['data']['train_generated']:
#            print("Attempting to load training data from file . . .")
#            try:
#                t_CORR_E = np.load('train_corr_list.npy').tolist()
#                for i in trainers:
#                    t_tatr.append(tatr_list[i])
#                break
#            except FileNotFoundError:
#                print("Data not found. Proceeding to data generation.")
#                inp['data']['train_generated'] = False
#        else:
#            print("Generating training set data . . .")
#            psi4.core.clean()
#            psi4.core.clean_options()
#            psi4.core.clean_variables()
#            psi4.set_options({"basis":bas})
#            for i in trainers:
#                t_tatr.append(tatr_list[i])
#                new_geom = geom.replace(gvar,str(pes[i]))
#                mol = psi4.geometry(new_geom)
#                _, wfn = psi4.energy(valtype,return_wfn=True)
#                t_CORR_E.append(wfn.variable('{} CORRELATION ENERGY'.format(valtype)))
#            np.save('train_corr_list.npy',t_CORR_E)
#            inp['data']['train_generated'] = True
#
#        # update the input
#        with open(inpf,'w') as f:
#            json.dump(inp, f, indent=4)
#
#        while inp['data']['valid_generated']:
#            print("Attempting to load validation data from file . . .")
#            try:
#                if ref == True:
#                    p_CORR_E.append(g_E_CORR_list[i])
#                    v_CORR_E = np.load('valid_corr_list.npy').tolist()
#                for i in vals:
#                    v_tatr.append(tatr_list[i])
#                    v_SCF.append(E_SCF_list[i])
#                    v_PES.append(pes[i])
#                break
#            except FileNotFoundError:
#                print("Data not found. Proceeding to data generation.")
#                inp['data']['grand_generated'] = False
#        else:
#            print("Generating validation set data . . .")
#            for i in vals:
#                v_tatr.append(tatr_list[i])
#                v_SCF.append(E_SCF_list[i])
#                v_PES.append(pes[i])
#                if ref == True:
#                    p_CORR_E.append(g_E_CORR_list[i])
#                    new_geom = geom.replace(gvar,str(pes[i]))
#                    mol = psi4.geometry(new_geom)
#                    _, wfn = psi4.energy(valtype,return_wfn=True)
#                    v_CORR_E.append(wfn.variable('{} CORRELATION ENERGY'.format(valtype)))
#            if ref == True:
#                np.save('valid_corr_list.npy',v_CORR_E)
#            inp['data']['valid_generated'] = True
#
#        # update the input
#        with open(inpf,'w') as f:
#            json.dump(inp, f, indent=4)
#
#    # internally shift all corr E by the avg training set corr E
#    t_CORR_avg = np.mean(t_CORR_E)
#    t_CORR_E = np.subtract(t_CORR_E,t_CORR_avg)
#    if ref == True:
#        v_CORR_E = np.subtract(v_CORR_E,t_CORR_avg)
#        p_CORR_E = np.subtract(p_CORR_E,t_CORR_avg)
#        # }}}
#
#    # model determination# {{{
#    # train the hypers
#    if inp['data']['hypers']:
#        print("Loading hyperparameters from {}".format(inpf))
#        s = inp['data']['s']
#        l = inp['data']['l']
#    else:
#        print("Determining hyperparameters via k-fold cross validation . . .")
#        s, l = log_space_cv(t_CORR_E,t_tatr,k=M)
#        inp['data']['hypers'] = True
#        inp['data']['s'] = s
#        inp['data']['l'] = l
#        with open(inpf,'w') as f:
#            json.dump(inp, f, indent=4)
#
#    # train for alpha
#    if inp['data']['a']:
#        print("Loading coefficients from {}".format(inpf)) 
#        alpha = np.asarray(inp['data']['a'])
#    else:
#        print("Model training using s = {} and l = {} . . .".format(s,l))
#        alpha = train(t_tatr,t_CORR_E,s,l)
#        inp['data']['a'] = alpha.tolist()
#        with open(inpf,'w') as f:
#            json.dump(inp, f, indent=4)# }}}
#
#    # predict E across the PES# {{{
#    print("Predicting PES . . .")
#    pred_E_list = pred(v_tatr,t_tatr,alpha,s,l)
#
#    pred_E_list = np.add(pred_E_list,t_CORR_avg)
#    pred_E_list = np.add(pred_E_list,v_SCF)
#    if ref == True:
#        v_E_list = np.add(v_CORR_E,t_CORR_avg)
#        v_E_list = np.add(v_E_list,v_SCF)
#        p_E_list = np.add(p_CORR_E,t_CORR_avg)
#        p_E_list = np.add(p_E_list,v_SCF)
#        # }}}
#
#    # plot# {{{
#    if inp['setup']['plot']:
#        if inp['setup']['dpi']:
#            plt.figure(1,dpi=inp['setup']['dpi'])
#        else:
#            plt.figure(1,dpi=100)
#        plt.plot(v_PES,v_SCF,'y-',label='SCF')
#        if ref == True:
#            plt.plot(v_PES,p_E_list,'r--',label='{} PES'.format(predtype))
#            plt.plot(v_PES,v_E_list,'b-o',label='{} PES'.format(valtype),linewidth=3)
#        plt.plot(v_PES,pred_E_list,'r-^',ms=2,label='{}/ML-{}'.format(predtype,M),linewidth=2)
#        plt.plot(t_pos,[-1.18 for i in range(0,len(t_pos))],'go',label='Training points')
#        if inp['setup']['axis']:
#            plt.axis(inp['setup']['axis'])
#        plt.xlabel('r/Angstrom')
#        plt.ylabel('Energy/$E_h$')
#        plt.legend()
#        if inp['setup']['ptype'] == "save":
#            plt.savefig('plot.png')
#        else:
#            plt.show()# }}}
#    # }}}

def k_means(pts,M):
    # {{{
    '''
    Cluster `pts` into `M` clusters by the k-means algorithm

    Parameters
    ----------
    pts: list of initial data points pts
    M: int number of centers desired

    Returns
    -------
    maps: list of clusters containing `(point,number)` tuples
    where `number` is the location of `point` in `pts`

    Notes
    -----
    The algorithm randomly selects M initial centers from `pts`. Then, 
    it iteratively re-clusters until convergence. The return is thus the 
    converged optimal  mapping of the clusters (see `cluster` return).
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

def k_means_loop(pts,M,K):
    # {{{
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

def cen_err(maps):
    # {{{
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
    return np.std(errors), close_pts
# }}}

def cluster(pts,M,cens):
    # {{{
    '''
    Form cluster map holding the point and the number from the original list
    Cluster M points by norm difference of points
    Return the cluster-point maps and the point-center differences

    Parameters
    ----------
    pts: list or numpy array of data point vectors
    M: int number of clusters desired
    cens: list or numpy array of vectors of cluster centers

    Returns
    -------
    maps: list of clusters containing `(point,number)` tuples
    where `number` is the location of `point` in `pts`

    Notes
    -----
    Map structure:
    [[cluster 1: (pt,#), . . .],[cluster 2: (pt,#), . . .]]
    '''
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

def recenter(M,maps):
    # {{{
    '''
    Return centers (vector means) of each of the M cluster in maps
    '''
    new_cens = []
    for cen in range(0,M):
        cluster = [x[0] for x in maps[cen]]
        new_cen = np.mean([x[0] for x in maps[cen]],axis=0)
        new_cens.append(new_cen)
    return np.asarray(new_cens)# }}}
