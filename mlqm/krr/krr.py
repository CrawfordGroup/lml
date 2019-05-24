import numpy as np
from numpy import linalg as la
import psi4
psi4.core.be_quiet()
import json
import matplotlib
import matplotlib.pyplot as plt
import random
import copy

def do_krr(inpf,save=True):# {{{
    '''
    inpf must contain AT LEAST:
    inp['mol']['geom']: a geometry with a variable (gvar) given for displacement along PES
    inp['setup']: 'M', 'N', 'st', 'K', 'gvar' (list), 'grange' (list), 'basis'
    See examples at github.com/bgpeyton/lml/mlqm/krr/examples/

    Kernel Ridge Regression with M training points and N total representations
    st used for tatr generation
    K repeated k-means algorithms, best clustering chosen
    save option to write TATRs and SCF/CCSD energies to file

    The algorithm will displace a diatomic over `grange[0]-grange[1]` Angstroms, generating
    TATRs and corresponding energies at each point. Then, a training set of `M` TATRs 
    will be selected by the k-means algorithm. Using this training set, the 
    hyperparameters `s` and `l` (kernel width and regularization) will be determined by 
    searching over a coarse logarithmic grid and minimizing the regularized L2 squared 
    loss cost function. Once the hypers are thus determined, the regression coefficients 
    `a` are trained by solving the corresponding linear equations. A validation set 
    spread across the PES (not including any training points) is then predicted using the 
    model, and the results are graphed.
    '''
    # parse input# {{{
    with open(inpf,'r') as f:
        inp = json.load(f)
    M    = inp['setup']['M']
    N    = inp['setup']['N']
    st   = inp['setup']['st']
    K    = inp['setup']['K']
    bas  = inp['setup']['basis']
    geom = inp['mol']['geom']# }}}

    # generate the total data set # {{{
    # N evenly-spaced representations on the PES
    bot = float(inp['setup']['grange'][0])
    top = float(inp['setup']['grange'][1])
    gvar = inp['setup']['gvar'][0]
    pes = np.linspace(bot,top,N) 
    while inp['data']['generated']:
        try:
            print("Loading data set . . .")
            tatr_list = np.load('t_list.npy').tolist()
            E_CCSD_CORR_list = np.load('corr_list.npy').tolist()
            E_SCF_list = np.load('scf_list.npy').tolist()
            break
        except FileNotFoundError:
            print("Data not found. Proceeding to data generation.")
            inp['data']['generated'] = False
    else:
        print("Generating data set . . .")
        tatr_list = [] # hold TATRs
        E_CCSD_CORR_list = [] # hold CCSD correlation energy
        E_SCF_list = [] # hold SCF energy

        for i in range(0,N):
            new_geom = geom.replace(gvar,str(pes[i]))
            mol = psi4.geometry(new_geom)
            tatr, wfn = make_tatr(mol,bas,st=st)
            tatr_list.append(tatr)
            E_CCSD_CORR_list.append(wfn.variable('CCSD CORRELATION ENERGY'))
            E_SCF_list.append(wfn.variable('SCF TOTAL ENERGY'))
        inp['data']['generated'] = True
        if save:
            np.save('t_list.npy',tatr_list)
            np.save('corr_list.npy',E_CCSD_CORR_list)
            np.save('scf_list.npy',E_SCF_list)
    # update the input
    with open(inpf,'w') as f:
        json.dump(inp, f, indent=4)# }}}

    # generate the training/validation sets# {{{
    if inp['data']['trainers']:
        print("Loading training set from {} . . .".format(inpf))
        trainers = inp['data']['trainers']
        if len(trainers) != M:
            print("Given training set does not have {} points! Re-generating . . .".format(M))
            trainers = gen_train(tatr_list, M, K) 
            inp['data']['trainers'] = trainers
            with open(inpf,'w') as f:
                json.dump(inp, f, indent=4)
    else:
        print("Generating training set . . .")
        trainers = gen_train(tatr_list, M, K) 
        inp['data']['trainers'] = trainers
        with open(inpf,'w') as f:
            json.dump(inp, f, indent=4)

    print("Training set: {}".format(trainers))
    t_pos = [pes[i] for i in trainers]

    vals = sorted([i for i in range(0,200,4)], reverse=True) # validation set
    vals = sorted(list(set(vals) - (set(vals) & set(trainers))),reverse=True) # pull trainers out
    print("Validation set: {}".format(vals))

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
        vtatr = tatr_list[i]
        vE = E_CCSD_CORR_list.pop(i)
        v_tatr.append(vtatr)
        v_CORR_E.append(vE)
        v_SCF.append(E_SCF_list[i])
        v_PES.append(pes[i])

    # internally shift all corr E by the avg training set corr E
    t_CORR_avg = np.mean(t_CORR_E)
    t_CORR_E = np.subtract(t_CORR_E,t_CORR_avg)
    v_CORR_E = np.subtract(v_CORR_E,t_CORR_avg)# }}}

    # model determination# {{{
    # train the hypers
    if inp['data']['hypers']:
        print("Loading hyperparameters from {}".format(inpf))
        s = inp['data']['s']
        l = inp['data']['l']
    else:
        print("Determining hyperparameters via k-fold cross validation . . .")
        s, l = log_space_cv(t_CORR_E,t_tatr,k=M)
        inp['data']['hypers'] = True
        inp['data']['s'] = s
        inp['data']['l'] = l
        with open(inpf,'w') as f:
            json.dump(inp, f, indent=4)

    # train for alpha
    if inp['data']['a']:
        print("Loading coefficients from {}".format(inpf)) 
        alpha = np.asarray(inp['data']['a'])
    else:
        print("Model training using s = {} and l = {} . . .".format(s,l))
        alpha = train(t_tatr,t_CORR_E,s,l)
        inp['data']['a'] = alpha.tolist()
        with open(inpf,'w') as f:
            json.dump(inp, f, indent=4)# }}}

    # predict E across the PES# {{{
    print("Predicting PES . . .")
    pred_E_list = pred(v_tatr,t_tatr,alpha,s,l)

    v_E_list = np.add(v_CORR_E,t_CORR_avg)
    v_E_list = np.add(v_E_list,v_SCF)
    pred_E_list = np.add(pred_E_list,t_CORR_avg)
    pred_E_list = np.add(pred_E_list,v_SCF)# }}}

    # plot# {{{
    if inp['setup']['plot']:
        plt.figure(1,dpi=200)
        plt.plot(v_PES,v_E_list,'b-o',label='CCSD PES',linewidth=3)
        plt.plot(v_PES,v_SCF,'y-',label='SCF')
        plt.plot(v_PES,pred_E_list,'r-^',ms=2,label='CCSD/ML-{}'.format(M),linewidth=2)
        plt.plot(t_pos,[-1.18 for i in range(0,len(t_pos))],'go',label='Training points')
        if inp['setup']['axis']:
            plt.axis(inp['setup']['axis'])
        plt.xlabel('r/Angstrom')
        plt.ylabel('Energy/$E_h$')
        plt.legend()
        if inp['setup']['ptype'] == "save":
            plt.savefig('plot.png')
        else:
            plt.show()# }}}
    # }}}

def gaus(x, u, s):# {{{
    '''
    return a gaussian centered on u with width s
    note: we're using x within [-1,1]
    '''
    return np.exp(-(x-u)**2 / (2.0*s**2))# }}}

def make_tatr(mol,bas,x=150,st=0.05,graph=False):# {{{
    '''
    make t-amp tensor representation
    pass in a psi4 molecule and basis
    plus (optional) the number of points
    and (optional) the sigma for the gaussian
    can also graph the 1TATR and 2TATR
    '''
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    psi4.set_options({
        'basis':bas,
        'scf_type':'pk',
        'freeze_core':'false',
        'e_convergence':1e-8,
        'd_convergence':1e-8})
    psi4.set_module_options('SCF',
        {'e_convergence': 1e-8, 'd_convergence': 1e-8,
         'DIIS': True, 'scf_type':'pk'})
    e, wfn = psi4.energy('ccsd',molecule=mol,return_wfn=True)
    amps = wfn.get_amplitudes()

    # sort amplitudes by magnitude (sorted(x,key=abs) will ignore sign) 
    t1 = sorted(amps['tIA'].to_array().ravel(),key=abs)[-x:]
    t2 = sorted(amps['tIjAb'].to_array().ravel(),key=abs)[-x:]

    tatr = [] # store eq vals
    tatr1 = [] # singles tatrs
    tatr2 = [] # doubles tatrs
    x_list = np.linspace(-1,1,x)
    for i in range(0,x):
        val1 = 0
        val2 = 0
        for t_1 in range(0,len(t1)):
            val1 += gaus(x_list[i],t1[t_1],st)
        for t_2 in range(0,len(t2)):
            val2 += gaus(x_list[i],t2[t_2],st)
        tatr1.append(val1)
        tatr2.append(val2)
        tatr.append(val1)
        tatr.append(val2)

    if graph:
        plt.figure(1)
        plt.plot(x_list,tatr1,label="singles")
        plt.plot(x_list,tatr2,label="doubles")
        plt.legend()
        plt.show()
    return np.asarray(tatr), wfn# }}}

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
    for i in range(0,N):
        Y.append(np.dot(a,k[i]))
    
    return Y# }}}

def loss(y,y_p,l,a):# {{{
    '''
    Calculate L2 (squared) loss with regularization to protect against
    large norm squared regression coefficients
    given true and predicted values, regularization, and regression coefficients
    '''
    ls = sum([(y_p[i] - y[i])**2 for i in range(len(y))]) + l*np.linalg.norm(a)**2
    return ls# }}}

def grid_search(tr_x,val_x,tr_y,val_y):# {{{
    mse = 1E9
    s_list = np.logspace(-5,8,num=16)
    l_list = np.logspace(-8,-1,num=16)
    for s in s_list:
        for l in l_list:
            a = train(tr_x,tr_y,s,l)
            y_p = pred(val_x,tr_x,a,s,l)
            new_mse = abs(loss(val_y,y_p,l,a))
            if new_mse <= mse:
                s_f = s
                l_f = l
                mse = new_mse
    return s_f, l_f, mse# }}}

def log_space_cv(y_data,x_data,k=5):# {{{
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
    return s,l # }}}

if __name__ == "__main__":
    import sys
    inp = str(sys.argv[1])
    do_krr(inp)
