import psi4
psi4.core.be_quiet()
import numpy as np

from . import molecule

"""
repgen

This module contains a series of functions to create a variety of
representations from molecular data.

Contributors: Ben Peyton, Connor Briggs

Functions
---------
make_coulomb
    Creates the Coulomb matrix based on the geometry.

make_tatr
    Creates the T-amplitude tensor representation from the T2 and
    optionally the T1 tensors.

make_dtr
    Creates the density tensor representation from the one-particle and
    optionally the two-particle density matrices.

gaus
    Calculate the value along a Gaussian curve.
"""

def make_coulomb(molecule, ignore_matrix_symmetry = True, **kwargs) :
    """
    make_coulomb
    
    Make a list of the biggest values in the Coulomb matrices for the
    molecules with an optional number of points and an option to ignore
    the symmetry of the matrix.

    Formula for the Coulomb matrix from
    https://singroup.github.io/dscribe/tutorials/coulomb_matrix.html

    Contributor: Connor Briggs

    Parameters
    ----------
    molecule
        the molecule.Molecule to use to create the reps.
        
    ignore_matrix_symmetry = True
        Whether to ignore the fact that the Coulomb matrix is symmetric when
        considering the elements. If False, then the off-diagonal elements
        will not be duplicated.

    cutoff = None, n = 100
        The absolute cutoff for which elements are considered significant
        (cutoff), and the number of elements to include (n). If cutoff is None,
        then the behavior depends on n. If n <= 0 or n == 'full'
        and cutoff is None, then all elements will be included. If n > 0, and
        cutoff is None, then the n most significant elements will be included.
        If n > 0 and cutoff is not None, then the n most significant elements
        above cutoff will be included, and if there are not enough elements
        above the cutoff, the rest will be zero-padded.
    """
    out = []
    charges = [g[1] for g in molecule.geometry()]
    coul = [[(charges[k][i] ** 2.4 / 2) if i == j else
                charges[k][i] * charges[k][j] /
             np.linalg.norm(np.array(coords[k][i]) -
                            np.array(coords[k][j]))
             for j in range(len(charges[k]))] for i in
            range(len(charges[k]))]

    if ignore_matrix_symmetry :
        for r in coul :
            out.extend(r)
    else :
        for i in range(len(coul)) :
            out.extend(coul[i][i:])
    out = sorted(out, reverse = True)
        
    if "cutoff" in kwargs and kwargs["cutoff"] != None :
        reps = [r for r in reps if abs(r) >= abs(kwargs["cutoff"])]
    if "n" in kwargs and not \
        ((kwargs["n"] is str and kwargs["n"].lower() == "full") or
        kwargs["n"] == None or kwargs["n"] <= 0) :
        if len(reps) < kwargs["n"] :
            reps.extend(0 for i in range(kwargs["n"] - len(reps)))
        else :
            reps = reps[0:kwargs["n"] - 1]
    return out

def make_tatr(method,t2,t1=None,x=150,st=0.05):
# {{{
    '''
    make_tatr
    
    Make the T-amplitude tensor representation.

    Contributor: Ben Peyton

    Parameters
    ----------
    method
        A string containing the method for the T amplitudes. Can
        be currently 'CCSD', 'CCSD-NAT', or 'MP2'.

    t2
        The T2 amplitudes.

    t1 = None
        The T1 amplitudes if desired.

    x = 150
        The number of significant representations to return.

    st = 0.05
        The width of the Gaussian.

    Raises
    ------
    ValueError
        If the method is not recognized.

    Returns
    -------
    The T-amplitude tensor representation.
    '''

    if (method.upper() == "CCSD") or (method.upper() == "CCSD-NAT"):
        # {{{
        # sort amplitudes by magnitude and keep the highest x
        # (sorted(x,key=abs) will ignore sign) 
        t1 = sorted(t1.ravel(),key=abs)[-x:]
        t2 = sorted(t2.ravel(),key=abs)[-x:]

        # make a discretized gaussian using the amps
        tatr = [] # store eq vals
        x_list = np.linspace(-1,1,x)
        for i in range(0,x):
            val1 = 0
            val2 = 0
            for t_1 in range(0,len(t1)):
                val1 += gaus(x_list[i],t1[t_1],st)
            for t_2 in range(0,len(t2)):
                val2 += gaus(x_list[i],t2[t_2],st)
            tatr.append(val1)
            tatr.append(val2)
        # }}}

    elif method.upper() == "MP2":
        # {{{
        # sort amplitudes by magnitude and keep the highest x
        # (sorted(x,key=abs) will ignore sign) 
        t2 = sorted(t2.ravel(),key=abs)[-x:]

        # make a discretized gaussian using the amps
        tatr = [] # store eq vals
        x_list = np.linspace(-1,1,x)
        for i in range(0,x):
            val2 = 0
            for t_2 in range(0,len(t2)):
                val2 += gaus(x_list[i],t2[t_2],st)
            tatr.append(val2)
        # }}}

    else: 
        raise ValueError("{method} amplitude representations not supported!")

    return np.asarray(tatr)
# }}}

def make_dtr(opdm,tpdm=None,x=150,st=0.05,cut_type='full',cut_val=None):
# {{{
    """
    make_dtr
    
    Many-body tensor with density elements (density tensor representation). 
    Currently tested for MP2 OPDM only.

    Contributor: Ben Peyton

    Parameters
    ----------
    opdm
        The one-particle density matrix.

    tpdm = None
        The two-particle density matrix, if desired.

    x = 150
        The number of points to use to discretize the tensor.

    st = 0.05
        The width of the gaussians to sum over when creating the reps.

    cut_type = 'full'
        The way to determine the number of reps to return. 'full' gives all
        the reps; 'min' gives any reps above cut_val; 'top' gives the largest
        reps, totaling cut_val reps; 'percent' gives the top cut_val percent of
        the total; 'percent_min' gives the top cut_val[0] percent of all reps
        above cut_val[1].

    cut_val = None
        The cutoff value. Its contents depend on cut_type. If cut_type is
        'full', then it should be None; if 'min' or 'percent' it should be
        a float; if 'top' it should be an int; if 'percent_min' it should
        be indexable containing two floats representing the fraction and the
        cutoff value.

    Raises
    ------
    ValueError
        if cut_type is not recognized.

    Returns
    -------
    The density tensor representation.
    """

    # sort OPDM and t2 by magnitude (sorted(x,key=abs) will ignore sign) 
    opdm = np.array(sorted(opdm.ravel(),key=abs))
    if tpdm:
        tpdm = np.array(sorted(tpdm.ravel(),key=abs))

    # deal with cutoffs
    if cut_type.lower() == 'full':
        pass
    elif cut_type.lower() == 'min':
        opdm = opdm[abs(opdm) > cut_val]
        if tpdm:
            tpdm = tpdm[abs(tpdm) > cut_val]
    elif cut_type.lower() == 'top':
        opdm = opdm[-cut_val:]
        if tpdm:
            tpdm = tpdm[-cut_val:]
    elif cut_type.lower() == 'percent':
        opdm = opdm[-round(cut_val*len(opdm)):]
        if tpdm:
            tpdm = tpdm[-round(cut_val*len(tpdm)):]
    elif cut_type.lower() == 'percent_min':
        opdm = opdm[abs(opdm) > cut_val[1]]
        opdm = opdm[-round(cut_val[0]*len(opdm)):]
        if tpdm:
            tpdm = tpdm[abs(tpdm) > cut_val[1]]
            tpdm = tpdm[-round(cut_val[0]*len(tpdm)):]
    else:
        raise ValueError("Cutoff type '{cut_type}' not recognized.")

    # make a discretized gaussian using the PDMs
    dtr = [] # store eq vals
    x_list = np.linspace(-1,1,x)

    # sum over d-centered gaussians for every x
    for i in range(0,x):
        val1 = sum([gaus(x_list[i],d,st) for d in opdm])
        dtr.append(val1)
        if tpdm:
            val2 = sum([gaus(x_list[i],d,st) for d in tpdm])
            dtr.append(val2)

    return np.asarray(dtr)
# }}}

def gaus(x, u, s):
# {{{
    '''
    gaus

    Returns the value of a Gaussian curve at point x centered on u with a width
    of s.

    Contributor: Ben Peyton

    Parameters
    ----------
    x
        The distance of the point from the origin.

    u
        The distance of the peak from the origin.

    s
        The width of the peak.

    Returns
    -------
    The value of the specified Gaussian curve.
    '''
    return np.exp(-1 * (x-u)**2 / (2.0 * s**2))
# }}}
