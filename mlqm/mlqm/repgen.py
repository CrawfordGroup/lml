import psi4
psi4.core.be_quiet()
import numpy as np

def make_tatr(mol,theory,bas,x=150,st=0.05,graph=False):
# {{{
    '''
    make t-amp tensor representation
    pass in a psi4 molecule with theory and basis
    plus (optional) the number of points
    and (optional) the sigma for the gaussian
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

    if theory == "CCSD":# {{{
        # compute and grab amplitudes
        e, wfn = psi4.energy("CCSD",molecule=mol,return_wfn=True)
        amps = wfn.get_amplitudes()

        # sort amplitudes by magnitude (sorted(x,key=abs) will ignore sign) 
        t1 = sorted(amps['tIA'].to_array().ravel(),key=abs)[-x:]
        t2 = sorted(amps['tIjAb'].to_array().ravel(),key=abs)[-x:]

        # make a discretized gaussian using the amps
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
            tatr.append(val2)# }}}

    elif theory == "MP2":# {{{
        # no python access to MP2 amps, compute them by hand
        # see spin-orbital CCSD code in Psi4Numpy
        e, wfn = psi4.energy("MP2",molecule=mol,return_wfn=True)
        mints = psi4.core.MintsHelper(wfn.basisset())
        nocc = wfn.doccpi()[0] * 2
        nvirt = wfn.nmo()*2 - nocc
        MO = np.asarray(mints.mo_spin_eri(wfn.Ca(), wfn.Ca()))
        o = slice(0, nocc)
        v = slice(nocc, MO.shape[0])
        H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
        H = np.einsum('uj,vi,uv', wfn.Ca(), wfn.Ca(), H)
        H = np.repeat(H, 2, axis=0)
        H = np.repeat(H, 2, axis=1)
        spin_ind = np.arange(H.shape[0], dtype=np.int) % 2
        H *= (spin_ind.reshape(-1, 1) == spin_ind)
        MOijab = MO[o, o, v, v]
        F = H + np.einsum('pmqm->pq', MO[:, o, :, o])
        Focc = F[np.arange(nocc), np.arange(nocc)].flatten()
        Fvirt = F[np.arange(nocc, nvirt + nocc), np.arange(nocc, nvirt + nocc)].flatten()
        Dijab = Focc.reshape(-1, 1, 1, 1) + Focc.reshape(-1, 1, 1) - Fvirt.reshape(-1, 1) - Fvirt
        t2_full = MOijab / Dijab

        # sort amplitudes by magnitude (sorted(x,key=abs) will ignore sign) 
        t2 = sorted(t2_full.ravel(),key=abs)[-x:]

        # make a discretized gaussian using the amps
        tatr = [] # store eq vals
        x_list = np.linspace(-1,1,x)
        for i in range(0,x):
            val2 = 0
            for t_2 in range(0,len(t2)):
                val2 += gaus(x_list[i],t2[t_2],st)
            tatr.append(val2)# }}}

    else: 
        print("I don't know how to handle {} amplitudes.".format(theory))
        raise Exception("{} amplitude representations not supported!")

    return np.asarray(tatr), wfn
# }}}

def gaus(x, u, s):
# {{{
    '''
    return a gaussian centered on u with width s
    note: we're using x within [-1,1]
    '''
    return np.exp(-(x-u)**2 / (2.0*s**2))
# }}}

