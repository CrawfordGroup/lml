import psi4
psi4.core.be_quiet()
import json
import numpy as np
import repgen

def pes_gen(ds):
# {{{
    '''
    Generate a PES and the representations across it.
    '''
    with open(ds.inpf,'r') as f:
        inp = json.load(f)
    # generate the grand data set representations
    # N evenly-spaced representations on the PES
    bot = float(inp['setup']['grange'][0])
    top = float(inp['setup']['grange'][1])
    gvar = inp['setup']['gvar'][0]
    pes = np.linspace(bot,top,ds.N) 

    while inp['data']['grand_generated']:
        try:
            print("Loading potential energy surface data set . . .")
            tatr_list = np.load('t_list.npy').tolist()
            E_SCF_list = np.load('scf_list.npy').tolist()
            if (ds.predtype == ds.valtype) or (ds.ref == True):
                g_E_CORR_list = np.load('grand_corr_list.npy').tolist() 
                # grand low-level correlation energy needed if pred=val, or if you want
                # the reference (low-level) energies to graph
            break
        except FileNotFoundError:
            print("Potential energy surface data not found. Proceeding to data generation.")
            inp['data']['grand_generated'] = False

    else:
        if ds.predtype in ["MP2","CCSD"]:
            print("Generating grand {} potential energy surface data set . . .".format(ds.predtype))
            tatr_list = [] # hold TATRs
            g_E_CORR_list = [] # hold low-level correlation energy for grand data set
            E_SCF_list = [] # hold SCF energy
            for i in range(0,ds.N):
                new_geom = ds.geom.replace(gvar,str(pes[i]))
                mol = psi4.geometry(new_geom)
                tatr, wfn = repgen.make_tatr(mol,ds.predtype,ds.bas,st=ds.st)
                tatr_list.append(tatr)
                g_E_CORR_list.append(wfn.variable('{} CORRELATION ENERGY'.format(ds.predtype)))
                E_SCF_list.append(wfn.variable('SCF TOTAL ENERGY'))
            inp['data']['grand_generated'] = True
            np.save('t_list.npy',tatr_list)
            np.save('grand_corr_list.npy',g_E_CORR_list)
            np.save('scf_list.npy',E_SCF_list)
        else:
            print("Prediction set type {} not supported for potential energy surfaces.".format(ds.predtype))
            raise Exception("Prediction set type {} not supported!".format(ds.predtype))

    # update the input
    with open(ds.inpf,'w') as f:
        json.dump(inp, f, indent=4)
    
    return tatr_list,g_E_CORR_list,E_SCF_list
        # }}}

def reg_l2(y,y_p,l,a):
    # {{{
    '''
    Calculate L2 (squared) loss with regularization to protect against
    large norm squared regression coefficients
    given true and predicted values, regularization, and regression coefficients
    '''
    ls = sum([(y_p[i] - y[i])**2 for i in range(len(y))]) + l*np.linalg.norm(a)**2
    return ls
# }}}
