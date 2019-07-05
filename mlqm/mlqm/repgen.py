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
        'mp2_type':'conv',
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
        x_list = np.linspace(-1,1,x)
        for i in range(0,x):
            val1 = 0
            val2 = 0
            for t_1 in range(0,len(t1)):
                val1 += gaus(x_list[i],t1[t_1],st)
            for t_2 in range(0,len(t2)):
                val2 += gaus(x_list[i],t2[t_2],st)
            tatr.append(val1)
            tatr.append(val2)# }}}

    elif theory == "CCSD-NAT":# {{{
        # compute and grab amplitudes
        scf_e,scf_wfn = psi4.energy('scf',return_wfn=True)
#        e,wfn1 = psi4.gradient('ccsd',return_wfn=True,ref_wfn=scf_wfn)
        e,wfn1 = psi4.energy('ccsd',molecule=mol,return_wfn=True,ref_wfn=scf_wfn)
        psi4.oeprop(wfn1,'DIPOLE')
        D = wfn1.Da_subset("MO").to_array()
        w,v = np.linalg.eigh(D)
        w = np.flip(w,axis=0)
        v = np.flip(v,axis=1)
        new_C = scf_wfn.Ca().to_array() @ v
        scf_wfn.Ca().copy(psi4.core.Matrix.from_array(new_C))
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.clean_variables()
        psi4.set_options({
            'basis':bas,
            'scf_type':'pk',
            'freeze_core':'false',
            'e_convergence':1e-8,
            'd_convergence':1e-8})
        e,wfn2 = psi4.energy('ccsd',molecule=mol,return_wfn=True,ref_wfn=scf_wfn)

        amps = wfn2.get_amplitudes()

        # sort amplitudes by magnitude (sorted(x,key=abs) will ignore sign) 
        t1 = sorted(amps['tIA'].to_array().ravel(),key=abs)[-x:]
        t2 = sorted(amps['tIjAb'].to_array().ravel(),key=abs)[-x:]

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
        return np.asarray(tatr), wfn2, wfn1
        # }}}

    elif theory == "MP2":# {{{
#        # no python access to MP2 amps, compute them by hand
#        # see spin-orbital CCSD code in Psi4Numpy
#        e, wfn = psi4.energy("MP2",molecule=mol,return_wfn=True)
#        mints = psi4.core.MintsHelper(wfn.basisset())
#        nocc = wfn.doccpi()[0] * 2
#        nvirt = wfn.nmo() * 2 - nocc
#        MO = np.asarray(mints.mo_spin_eri(wfn.Ca(), wfn.Ca()))
#        o = slice(0, nocc)
#        v = slice(nocc, MO.shape[0])
#        H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
#        H = np.einsum('uj,vi,uv', wfn.Ca(), wfn.Ca(), H)
#        H = np.repeat(H, 2, axis=0)
#        H = np.repeat(H, 2, axis=1)
#        spin_ind = np.arange(H.shape[0], dtype=np.int) % 2
#        H *= (spin_ind.reshape(-1, 1) == spin_ind)
#        MOijab = MO[o, o, v, v]
#        F = H + np.einsum('pmqm->pq', MO[:, o, :, o])
#        Focc = F[np.arange(nocc), np.arange(nocc)].flatten()
#        Fvirt = F[np.arange(nocc, nvirt + nocc), np.arange(nocc, nvirt + nocc)].flatten()
#        Dijab = Focc.reshape(-1, 1, 1, 1) + Focc.reshape(-1, 1, 1) - Fvirt.reshape(-1, 1) - Fvirt
#        t2_full = MOijab / Dijab

        # Perform MP2 Energy Calculation
        mp2_e, wfn = psi4.energy('MP2', return_wfn=True)
        
        # Relevant Variables
        natoms = mol.natom()
        nmo = wfn.nmo()
        nocc = wfn.doccpi()[0]
        nvir = nmo - nocc
        
        # MO Coefficients
        C = wfn.Ca_subset("AO", "ALL")
        npC = psi4.core.Matrix.to_array(C)
        
        # Integral generation from Psi4's MintsHelper
        mints = psi4.core.MintsHelper(wfn.basisset())
        
        # Build T, V, and S
        T = mints.ao_kinetic()
        npT = psi4.core.Matrix.to_array(T)
        V = mints.ao_potential()
        npV = psi4.core.Matrix.to_array(V)
        S = mints.ao_overlap()
        npS = psi4.core.Matrix.to_array(S)
        
        # Build ERIs
        ERI = mints.mo_eri(C, C, C, C)
        npERI = psi4.core.Matrix.to_array(ERI)
        # Physicist notation
        npERI = npERI.swapaxes(1, 2)
        
        # Build Core Hamiltonian in AO basis
        H_core = npT + npV
        
        # Transform H to MO basis
        H = np.einsum('uj,vi,uv', npC, npC, H_core, optimize=True)
        
        # Build Fock Matrix
        F = H + 2.0 * np.einsum('pmqm->pq', npERI[:, :nocc, :, :nocc], optimize=True)
        F -= np.einsum('pmmq->pq', npERI[:, :nocc, :nocc, :], optimize=True)
        
        # Occupied and Virtual Orbital Energies
        F_occ = np.diag(F)[:nocc]
        F_vir = np.diag(F)[nocc:nmo]
        
        # Build Denominator
        Dijab = F_occ.reshape(-1, 1, 1, 1) + F_occ.reshape(-1, 1, 1) - F_vir.reshape(
            -1, 1) - F_vir
        
        # Build T2 Amplitudes,
        # where t2 = <ij|ab> / (e_i + e_j - e_a - e_b),
        t2_full = npERI[:nocc, :nocc, nocc:, nocc:] / Dijab

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

def make_dtr(mol,theory,bas,x=300,st=0.05,graph=False):
# {{{
    if theory == "MP2":
        # set psi4 options
        # {{{
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.clean_variables()
        psi4.set_options({
            'basis':bas,
            'scf_type':'pk',
            'mp2_type':'conv',
            'freeze_core':'false',
            'e_convergence':1e-8,
            'd_convergence':1e-8})
        # }}}

        # Build MP2 one- and two-particle density matrices
        # see MP2 Gradient code in Psi4Numpy for details

        # start with the amplitudes, not available from Psi4 wfn object
        # {{{
        # Perform MP2 Energy Calculation
        mp2_e, wfn = psi4.energy('MP2', return_wfn=True)
        
        # Relevant Variables
        natoms = mol.natom()
        nmo = wfn.nmo()
        nocc = wfn.doccpi()[0]
        nvir = nmo - nocc
        
        # MO Coefficients
        C = wfn.Ca_subset("AO", "ALL")
        npC = psi4.core.Matrix.to_array(C)
        
        # Integral generation from Psi4's MintsHelper
        mints = psi4.core.MintsHelper(wfn.basisset())
        
        # Build T, V, and S
        T = mints.ao_kinetic()
        npT = psi4.core.Matrix.to_array(T)
        V = mints.ao_potential()
        npV = psi4.core.Matrix.to_array(V)
        S = mints.ao_overlap()
        npS = psi4.core.Matrix.to_array(S)
        
        # Build ERIs
        ERI = mints.mo_eri(C, C, C, C)
        npERI = psi4.core.Matrix.to_array(ERI)
        # Physicist notation
        npERI = npERI.swapaxes(1, 2)
        
        # Build Core Hamiltonian in AO basis
        H_core = npT + npV
        
        # Transform H to MO basis
        H = np.einsum('uj,vi,uv', npC, npC, H_core, optimize=True)
        
        # Build Fock Matrix
        F = H + 2.0 * np.einsum('pmqm->pq', npERI[:, :nocc, :, :nocc], optimize=True)
        F -= np.einsum('pmmq->pq', npERI[:, :nocc, :nocc, :], optimize=True)
        
        # Occupied and Virtual Orbital Energies
        F_occ = np.diag(F)[:nocc]
        F_vir = np.diag(F)[nocc:nmo]
        
        # Build Denominator
        Dijab = F_occ.reshape(-1, 1, 1, 1) + F_occ.reshape(-1, 1, 1) - F_vir.reshape(
            -1, 1) - F_vir
        
        # Build T2 Amplitudes,
        # where t2 = <ij|ab> / (e_i + e_j - e_a - e_b),
        t2 = npERI[:nocc, :nocc, nocc:, nocc:] / Dijab
        
        # Build T2_tilde Amplitudes (closed-shell spin-free analog of antisymmetrizer),
        # i.e., t2_tilde[p,q,r,s] = 2 * t2[p,q,r,s] - t2[p,q,s,r]),
        # where t2_tilde = [2<ij|ab> - <ij|ba>] / (e_i + e_j - e_a - e_b)
        t2_tilde = 2 * t2 - t2.swapaxes(2, 3)
        # }}}

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
#        Ppq = sorted(Ppq.ravel(),key=abs)[-x:]
#        Ppqrs = sorted(Ppqrs.ravel(),key=abs)[-x:]
        Ppq = sorted(Ppq.ravel())[-x:]
        Ppqrs = sorted(Ppqrs.ravel())[-x:]

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
        raise RuntimeError("DTR not supported for {}".format(theory))

    return np.asarray(dtr), wfn
# }}}

def gaus(x, u, s):
# {{{
    '''
    return a gaussian centered on u with width s
    note: we're using x within [-1,1]
    '''
    return np.exp(-(x-u)**2 / (2.0*s**2))
# }}}

