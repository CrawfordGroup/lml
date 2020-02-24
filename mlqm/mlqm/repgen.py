import psi4
psi4.core.be_quiet()
import numpy as np

def make_coulomb(coords, charges, ignore_matrix_symmetry = True, sort = False,
                 **kwargs) :
    """
    Returns the Coulomb matrix. When given options, the output is the flattened matrix. Passing in sort will sort the values
    so that the biggest is first. Passing in ignore_matrix_symmetry as False will cause the return value to be the flattened
    triangular matrix, thus removing duplicates. Passing in n cuts off the lowest values in the matrix until the total is 
    n. Passing in cutoff will cause the array to remove all values less than that cutoff. Passing in both will cut off the 
    greater number of indices from cutting off at n and cutting off values less than cutoff. These two options require sort
    to be True.

    Default values:
    ignore_matrix_symmetry = True
    n = "full"
    cutoff = None
    sort = False
    
    Formula for the Coulomb matrix from
    https://singroup.github.io/dscribe/tutorials/coulomb_matrix.html
    """
    
    coul = np.array([[(charges[i] ** 2.4 / 2) if i == j else
                 charges[i] * charges[j] /
                 np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
                 for j in range(len(charges))] for i in range(len(charges))])
    if ignore_matrix_symmetry and not sort :
        return coul
    if ignore_matrix_symmetry :
        reps = coul.flatten()
        if sort :
            reps = sorted(reps, reverse = True)
            if "cutoff" in kwargs and kwargs["cutoff"] != None :
                reps = [r for r in reps if abs(r) >= abs(kwargs["cutoff"])]
            if "n" in kwargs and not \
               ((kwargs["n"] is str and kwargs["n"].lower() == "full") or
                kwargs["n"] == None or kwargs["n"] <= 0) :
                if len(reps) < kwargs["n"] :
                    reps.extend(0 for i in range(kwargs["n"] - len(reps)))
                else :
                    reps = reps[0:kwargs["n"] - 1]    
        return reps
    else :
        reps = coul[np.tril_indices(len(coul))]
        if sort :
            reps = sorted(reps, reverse = True)
            if "cutoff" in kwargs and kwargs["cutoff"] != None :
                reps = [r for r in reps if abs(r) >= abs(kwargs["cutoff"])]
            if "n" in kwargs and not \
                ((kwargs["n"] is str and kwargs["n"].lower() == "full") or
                kwargs["n"] == None or kwargs["n"] <= 0) :
                if len(reps) < kwargs["n"] :
                    reps.extend(0 for i in range(kwargs["n"] - len(reps)))
                else :
                    reps = reps[0:kwargs["n"] - 1]
        return reps
    
def make_tatr(method,t,x=150,st=0.05,cut_type='top',cut_val=150):
# {{{
    '''
    Many-body tensor with wave function amplitudes (t-amplitude tensor representation). 
    method: string, 'ccsd' (requires 't2' and 't1' in t) or 'mp2' (requires only 't2')
    t: dictionary containing amplitudes
    x: points in discretization region along [-1:1] of the tensor (default 150)
    st: width of gaussians summed along the tensor (default 0.05)
    cut_type: type of cutoff of PDM elements after ravel and magnitude sort
        'full': use all elements (ignore cut_val)
        'min': use all elements above float(cut_val)
        'top': use highest int(cut_val) elements
        'percent': use highest float(cut_val)*100% elements
        'percent_min': tuple(m,n) highest float(m)*100% elements with magnitude greater than float(n)
    cut_val: value used in above cut_type (ignored if cut_type == 'full', type must match)
    '''

    if (method.upper() == "CCSD") or (method.upper() == "CCSD-NAT"):
        # {{{
        # sort amplitudes by magnitude
        # (sorted(x,key=abs) will ignore sign) 
        try:
            t1 = sorted(t['t1'].ravel(),key=abs)
        except KeyError:
            print('t1 amplitudes not found! CCSD-TATR generation halted.')
        try:
            t2 = sorted(t['t2'].ravel(),key=abs)
        except KeyError:
            print('t2 amplitudes not found! CCSD-TATR generation halted.')

        # deal with cutoffs
        if cut_type.lower() == 'full':
            pass
        elif cut_type.lower() == 'min':
            t1 = t1[abs(t1) > cut_val]
            t2 = t2[abs(t2) > cut_val]
        elif cut_type.lower() == 'top':
            t1 = t1[-cut_val:]
            t2 = t2[-cut_val:]
        elif cut_type.lower() == 'percent':
            t1 = t1[-round(cut_val*len(t1)):]
            t2 = t2[-round(cut_val*len(t2)):]
        elif cut_type.lower() == 'percent_min':
            t1 = t1[abs(t1) > cut_val[1]]
            t1 = t1[-round(cut_val[0]*len(t1)):]
            t2 = t2[abs(t2) > cut_val[1]]
            t2 = t2[-round(cut_val[0]*len(t2)):]
        else:
            raise RuntimeError("Cutoff type '{}' not recognized.".format(cut_type))

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
        # sort amplitudes by magnitude
        # (sorted(x,key=abs) will ignore sign) 
        try:
            t2 = sorted(t['t2'].ravel(),key=abs)
        except KeyError:
            print('t2 amplitudes not found! CCSD-TATR generation halted.')

        # deal with cutoffs
        if cut_type.lower() == 'full':
            pass
        elif cut_type.lower() == 'min':
            t2 = t2[abs(t2) > cut_val]
        elif cut_type.lower() == 'top':
            t2 = t2[-cut_val:]
        elif cut_type.lower() == 'percent':
            t2 = t2[-round(cut_val*len(t2)):]
        elif cut_type.lower() == 'percent_min':
            t2 = t2[abs(t2) > cut_val[1]]
            t2 = t2[-round(cut_val[0]*len(t2)):]
        else:
            raise RuntimeError("Cutoff type '{}' not recognized.".format(cut_type))

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
        print("I don't know how to handle {} amplitudes.".format(method))
        raise Exception("{} amplitude representations not supported!".format(method))

    return np.asarray(tatr)
# }}}

def make_dtr(opdm,tpdm=None,x=150,st=0.05,cut_type='full',cut_val=None):
# {{{
    """
    Many-body tensor with density elements (density tensor representation). 
    Currently tested for MP2 OPDM only.
    Pass in the OPDM given by wfn.Da() plus optional TPDM.
    x: points in discretization region along [-1:1] of the tensor (default 150)
    st: width of gaussians summed along the tensor (default 0.05)
    cut_type: type of cutoff of PDM elements after ravel and magnitude sort
        'full': use all elements (ignore cut_val)
        'min': use all elements above float(cut_val)
        'top': use highest int(cut_val) elements
        'percent': use highest float(cut_val)*100% elements
        'percent_min': tuple(m,n) highest float(m)*100% elements with magnitude greater than float(n)
    cut_val: value used in above cut_type (ignored if cut_type == 'full', type must match)
    """

    # sort OPDM and TPDM by magnitude (sorted(x,key=abs) will ignore sign) 
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
        raise RuntimeError("Cutoff type '{}' not recognized.".format(cut_type))

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
    return a gaussian point x centered on u with width s
    '''
    return np.exp(-1 * (x-u)**2 / (2.0*s**2))
# }}}

def legacy_CCSD_NAT():
#    elif theory == "CCSD-NAT":
#        # {{{
#        # compute and grab amplitudes
#        scf_e,scf_wfn = psi4.energy('scf',return_wfn=True)
##        e,wfn1 = psi4.gradient('ccsd',return_wfn=True,ref_wfn=scf_wfn)
#        e,wfn1 = psi4.energy('ccsd',molecule=mol,return_wfn=True,ref_wfn=scf_wfn)
#        psi4.oeprop(wfn1,'DIPOLE')
#        D = wfn1.Da_subset("MO").to_array()
#        w,v = np.linalg.eigh(D)
#        w = np.flip(w,axis=0)
#        v = np.flip(v,axis=1)
#        new_C = scf_wfn.Ca().to_array() @ v
#        scf_wfn.Ca().copy(psi4.core.Matrix.from_array(new_C))
#        psi4.core.clean()
#        psi4.core.clean_options()
#        psi4.core.clean_variables()
#        psi4.set_options({
#            'basis':bas,
#            'scf_type':'pk',
#            'freeze_core':'false',
#            'e_convergence':1e-8,
#            'd_convergence':1e-8})
#        e,wfn2 = psi4.energy('ccsd',molecule=mol,return_wfn=True,ref_wfn=scf_wfn)
#
#        amps = wfn2.get_amplitudes()
#
#        # sort amplitudes by magnitude (sorted(x,key=abs) will ignore sign) 
#        t1 = sorted(amps['tIA'].to_array().ravel(),key=abs)[-x:]
#        t2 = sorted(amps['tIjAb'].to_array().ravel(),key=abs)[-x:]
#
#        # make a discretized gaussian using the amps
#        tatr = [] # store eq vals
#        x_list = np.linspace(-1,1,x)
#        for i in range(0,x):
#            val1 = 0
#            val2 = 0
#            for t_1 in range(0,len(t1)):
#                val1 += gaus(x_list[i],t1[t_1],st)
#            for t_2 in range(0,len(t2)):
#                val2 += gaus(x_list[i],t2[t_2],st)
#            tatr.append(val1)
#            tatr.append(val2)
#        return np.asarray(tatr), wfn2, wfn1
#        # }}}
    pass

def legacy_make_dtr(method,t2,nmo,nocc,t1=None,x=300,st=0.05):
# {{{
    if method == "MP2":
        # Build T2_tilde Amplitudes (closed-shell spin-free analog of antisymmetrizer),
        # i.e., t2_tilde[p,q,r,s] = 2 * t2[p,q,r,s] - t2[p,q,s,r]),
        # where t2_tilde = [2<ij|ab> - <ij|ba>] / (e_i + e_j - e_a - e_b)
        t2_tilde = 2 * t2 - t2.swapaxes(2, 3)

        # Build MP2 one- and two-particle density matrices
        # see MP2 Gradient code in Psi4Numpy for details

        # Build MP2 OPDM
        # {{{
        Ppq = np.zeros((nmo, nmo))
        
        # Build OO block of MP2 OPDM
        # Pij = - 1/2 sum_kab [( t2(i,k,a,b) * t2_tilde(j,k,a,b) ) + ( t2(j,k,a,b) * t2_tilde(i,k,a,b) )]
        Pij = -0.5 * np.einsum('ikab,jkab->ij', t2, t2_tilde, optimize=True)
        Pij += -0.5 * np.einsum('jkab,ikab->ij', t2, t2_tilde, optimize=True)
        
        # Build VV block of MP2 OPDM
        # Pab = 1/2 sum_ijc [( t2(i,j,a,c) * t2_tilde(i,j,b,c) ) + ( t2(i,j,b,c) * t2_tilde(i,j,a,c) )]
        Pab = 0.5 * np.einsum('ijac,ijbc->ab', t2, t2_tilde, optimize=True)
        Pab += 0.5 * np.einsum('ijbc,ijac->ab', t2, t2_tilde, optimize=True)
        
        # Build Total OPDM
        Ppq[:nocc, :nocc] = Pij
        Ppq[nocc:, nocc:] = Pab
        # }}}

        # Build MP2 TPDM
        # {{{
        Ppqrs = np.zeros((nmo, nmo, nmo, nmo))
        
        # Build <OO|VV> and <VV|OO> blocks of MP2 TPDM
        Ppqrs[:nocc, :nocc, nocc:, nocc:] = t2
        Ppqrs[nocc:, nocc:, :nocc, :nocc] = t2.T
        # }}}

        # Build the density tensor representation
        # {{{
        # sort PDMs by magnitude (sorted(x,key=abs) will ignore sign) 
        Ppq = sorted(Ppq.ravel(),key=abs)[-x:]
        Ppqrs = sorted(Ppqrs.ravel(),key=abs)[-x:]

        # make a discretized gaussian using the PDMs
        dtr = [] # store eq vals
        x_list = np.linspace(-1,1,x)
        for i in range(0,x):
            val1 = 0
            val2 = 0
            for p_1 in range(0,len(Ppq)):
                val1 += gaus(x_list[i],Ppq[p_1],st)
            for p_2 in range(0,len(Ppqrs)):
                val2 += gaus(x_list[i],Ppqrs[p_2],st)
            dtr.append(val1)
            dtr.append(val2)
        # }}}

    else:
        print("Density tensor representations not available for this method.")
        raise RuntimeError("DTR not supported for {}".format(method))

    return np.asarray(dtr)
# }}}
