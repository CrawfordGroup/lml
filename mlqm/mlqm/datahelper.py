import psi4
psi4.core.be_quiet()
import json
import numpy as np
from . import repgen

def pes_gen(ds, **kwargs):
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

    if 'remove' in kwargs:
        pes = np.delete(pes,kwargs['remove'],axis=0)
        print("NOTE: PES is altered!")
        print("PES being generated with {} points.".format(len(pes)))

    if 'reptype' in kwargs:
        reptype = kwargs['reptype'].upper()
    else:
        reptype = 'TATR'

    print("Retrieving PES using {} representation . . .".format(reptype))

    while inp['data']['grand_generated']:
        try:
            print("Loading potential energy surface data set.")
            rep_list = np.load('rep_list.npy').tolist()
            E_SCF_list = np.load('ref_list.npy').tolist()
            if (ds.predtype == ds.valtype) or (ds.ref == True):
                g_E_CORR_list = np.load('grand_corr_list.npy').tolist() 
                # grand low-level correlation energy needed if pred=val, or if you want
                # the reference (low-level) energies to graph
            break
        except FileNotFoundError:
            print("Potential energy surface data not found. Proceeding to data generation.")
            inp['data']['grand_generated'] = False

    else:
        if ds.predtype in ["MP2","CCSD","CCSD-NAT"]:
            print("Generating grand {} potential energy surface data set . . .".format(ds.predtype))
            rep_list = [] # hold representations
            g_E_CORR_list = [] # hold low-level correlation energy for grand data set
            E_SCF_list = [] # hold SCF energy
            ref2_list = [] # hold reference (non-natural orbital) energies if needed
            for i in range(0,len(pes)):
                new_geom = ds.geom.replace(gvar,str(pes[i]))
                mol = psi4.geometry(new_geom)
                # TODO: this should call the appropriate repgen function
                # would like to be able to make TATRs, density representation, etc
                if ds.predtype == "CCSD-NAT":
                    # TODO: too hacky, but works for testing
                    rep, wfn, wfn1 = repgen.make_tatr(mol,ds.predtype,ds.bas,st=ds.st)
                    ref2_list.append(wfn1.variable('{} CORRELATION ENERGY'.format('CCSD')))
                    g_E_CORR_list.append(wfn.variable('{} CORRELATION ENERGY'.format('CCSD')))
                else:
                    if reptype == 'TATR':
                        rep, wfn = repgen.make_tatr(mol,ds.predtype,ds.bas,st=ds.st)
                    elif reptype == 'DTR':
                        rep, wfn = repgen.make_dtr(mol,ds.predtype,ds.bas,st=ds.st)
                    g_E_CORR_list.append(wfn.variable('{} CORRELATION ENERGY'.format(ds.predtype)))
                rep_list.append(rep)
                E_SCF_list.append(wfn.variable('SCF TOTAL ENERGY'))
            inp['data']['grand_generated'] = True
            np.save('rep_list.npy',rep_list)
            np.save('grand_corr_list.npy',g_E_CORR_list)
            np.save('ref_list.npy',E_SCF_list)
            if ds.predtype == "CCSD-NAT":
                np.save('ref2_list.npy',ref2_list)
        else:
            print("Prediction set type {} not supported for potential energy surfaces.".format(ds.predtype))
            raise Exception("Prediction set type {} not supported!".format(ds.predtype))

    # update the input
    with open(ds.inpf,'w') as f:
        json.dump(inp, f, indent=4)
    
    return rep_list,g_E_CORR_list,E_SCF_list
        # }}}

def data_gen(ds,pts,method):
# {{{
    '''
    Generate energies across a range of displacements, pts
    This is mostly just to generate extra data points 
    when doing Margraf-Reuter-type KRR across a PES. 
    '''
    with open(ds.inpf,'r') as f:
        inp = json.load(f)
    gvar = inp['setup']['gvar'][0]
    corr_E = []

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    psi4.set_options({"basis":ds.bas})
    for i in pts:
        new_geom = ds.geom.replace(gvar,str(i))
        mol = psi4.geometry(new_geom)
        _, wfn = psi4.energy(method,return_wfn=True)
        corr_E.append(wfn.variable('{} CORRELATION ENERGY'.format(method)))
    return corr_E
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

def grid_search(tr_x,val_x,tr_y,val_y):
# TODO: it would be great to have a GENERAL grid_search function which takes
# a prediction function, a training function, and a loss function w/ an
# arbitrary number of hyperparameters. Will keep this here as a placeholder
# {{{
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
    return s_f, l_f, mse
# }}}
