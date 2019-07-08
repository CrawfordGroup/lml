import psi4
psi4.core.be_quiet()
import numpy as np

def make_tatr(method,t2,t1=None,x=150,st=0.05):
# {{{
    '''
    make t-amp tensor representation and method
    pass in a dictionary containing the amplitudes
    plus (optional) the number of points
    and (optional) the sigma for the gaussian
    return the tatr
    '''

    if (method == "CCSD") or (method == "CCSD-NAT"):
        # {{{
        # sort amplitudes by magnitude and keep the highest x
        # (sorted(x,key=abs) will ignore sign) 
        t1 = sorted(t1.to_array().ravel(),key=abs)[-x:]
        t2 = sorted(t2.to_array().ravel(),key=abs)[-x:]

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

    elif method == "MP2":
        # {{{
        # sort amplitudes by magnitude (sorted(x,key=abs) will ignore sign) 
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
        raise Exception("{} amplitude representations not supported!")

    return np.asarray(tatr)
# }}}

def gaus(x, u, s):
# {{{
    '''
    return a gaussian centered on u with width s
    note: we're using x within [-1,1]
    '''
    return np.exp(-(x-u)**2 / (2.0*s**2))
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
