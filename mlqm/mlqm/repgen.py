import psi4
psi4.core.be_quiet()
import numpy as np

def make_coulomb(coords, charges, ignore_matrix_symmetry = True, **kwargs) :
    """
    Make a list of the biggest values in the Coulomb matrices for the
    molecules with an optional number of points and an option to ignore
    the symmetry of the matrix. The coords should be a list of arrays
    containing the 3-D coordinates of each atom. The charges should be
    the corresponding charge for each atom, so that charges[i] should be
    the charge on the atom at position coords[i]. n is the maximum number
    of reps to take. If n is "full" (case insensitive), 0, or None, this
    will return all the values, rather than cutting off. cutoff is the
    numeric cutoff for the values. If cutoff is set and n is set but not 0,
    None, or "full", then the function will return up to n values all above
    the cutoff. If ignore_matrix_symmetry is set to true, as it is by default,
    then the fact that the coulomb matrix is symmetric will be ignored. This
    means that all except the diagonal elements will be included twice. If it
    is set to false, then the symmetry is taken into account.

    Default values:
    ignore_matrix_symmetry = True
    n = 100
    cutoff = None

    Formula for the Coulomb matrix from
    https://singroup.github.io/dscribe/tutorials/coulomb_matrix.html
    """
    out = []
    for k in range(len(charges)) :
        coul = [[(charges[k][i] ** 2.4 / 2) if i == j else
                 charges[k][i] * charges[k][j] /
                 np.linalg.norm(np.array(coords[k][i]) - np.array(coords[k][j]))
                 for j in range(len(charges[k]))] for i in range(len(charges[k]))]
                                 
        if ignore_matrix_symmetry :
            reps = []
            for r in coul :
               reps.extend(r)
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
            elif "n" not in kwargs and "cutoff" not in kwargs :
            #If n is not passed and cutoff is not passed, default to 100 reps
                if len(reps) < 100 :
                    reps.extend(0 for i in range(100 - len(reps)))
                else :
                    reps = reps[0:99]
            out.append(reps)
        else :
            reps = []
            for i in range(len(coul)) :
                reps.extend(coul[i][i:])
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
            out.append(reps)
    return out

def make_tatr(method,t2,t1=None,x=150,st=0.05):
# {{{
    '''
    make t-amp tensor representation and method
    pass in a dictionary containing the amplitudes
    plus (optional) the number of points
    and (optional) the sigma for the gaussian
    return the tatr
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
        print("I don't know how to handle {} amplitudes.".format(method))
        raise Exception("{} amplitude representations not supported!".format(method))

    return np.asarray(tatr)
# }}}

def make_dtr(opdm,t2,x=150,st=0.05):
# {{{
    """
    Making a DTR. Currently implemented for MP2 only.
    Pass in the OPDM given by wfn.Da() plus any T2
    amplitudes (as the TPDM is simply amplitudes)
    """

    # sort OPDM and t2 by magnitude (sorted(x,key=abs) will ignore sign) 
    # subbing in t2 for TPDM
    opdm = sorted(opdm.ravel(),key=abs)[-x:]
    tpdm = sorted(t2.ravel(),key=abs)[-x:]

    # make a discretized gaussian using the PDMs
    dtr = [] # store eq vals
    x_list = np.linspace(-1,1,x)
    for i in range(0,x):
        val1 = 0
        val2 = 0
        for p_1 in range(0,len(opdm)):
            val1 += gaus(x_list[i],opdm[p_1],st)
        for p_2 in range(0,len(tpdm)):
            val2 += gaus(x_list[i],tpdm[p_2],st)
        dtr.append(val1)
        dtr.append(val2)

    return np.asarray(dtr)
# }}}

def gaus(x, u, s):
# {{{
    '''
    return a gaussian centered on u with width s
    note: we're using x within [-1,1]
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
