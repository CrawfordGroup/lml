import random
import numpy as np
from numpy import linalg as la
from . import datahelper

def k_means(pts,M,**kwargs):
    # {{{
    '''
    Cluster `pts` into `M` clusters by the k-means algorithm

    Parameters
    ----------
    pts: list of initial data points
    M: int number of centers desired

    Returns
    -------
    maps: list of clusters containing `(point,number)` tuples
    where `number` is the location of `point` in `pts`

    Notes
    -----
    The algorithm randomly selects M initial centers from `pts`. Then, 
    it iteratively re-clusters until convergence. The return is thus the 
    converged optimal mapping of the clusters (see `cluster` return).
    Map structure:
    [[cluster 1: (pt,#), . . .],[cluster 2: (pt,#), . . .]]
    '''
    if 'remove' in kwargs:
        pts = np.delete(pts,kwargs['remove'],axis=0)

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

def k_means_loop(pts,M,K,**kwargs):
    # {{{
    '''
    Loop the k-means algorithm K times
    Return the best clustered map and position of closest points to centers
    '''
    print("Looping the k-means algorithm {} times to find {} optimum points.".format(K,M))
    maps_list = [] # hold all the maps
    errors = [] # hold all the errors
    close_pts_list = [] # hold all the closest points
    for i in range(0,K):
        maps = k_means(pts,M,**kwargs)
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
    min_diff = -1

    errors = []
    close_pts = []
    for i in range(0,len(maps)): # loop over centers
        cen = np.mean([c[0] for c in maps[i]],axis=0) # mean of coords in center i
        diffs = [] # hold diffs for center i
        for j in range(0,len(maps[i])): # loop over points

#            diffs.append(la.norm(maps[i][j][0] - cen)) # norm of diff btwn point and cen
#        errors.append(np.mean(diffs)) # store the mean error for center i
#        close_pts.append(np.argmin(diffs)) # store the position of the closest point to center i
#    return np.std(errors), close_pts


            diff = -2 * np.dot(maps[i][j][0],cen)
            diff += la.norm(maps[i][j][0]) ** 2.0
            diff += la.norm(cen) ** 2.0
            diffs.append(diff)
            if min_diff == -1 or diff < min_diff:
                min_diff = diff
        errors.append(min_diff) # store the lowest diff for center i 
        close_pts.append(np.argmin(diffs)) # store the position of the closest point to center i
    return sum(errors), close_pts
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
