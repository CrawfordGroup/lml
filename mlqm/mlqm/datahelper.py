import psi4
psi4.core.be_quiet()
import json
import numpy as np
import os
import subprocess
from . import repgen

def write_psi4_input(molstr,method,global_options,**kwargs):
    # {{{
    '''
    Pass in a molecule string, method, and global options dictionary
    Kwargs to hold optional directory, module options dictionary,
    alternative calls (ie properties) and extra commands as strings 
    Writes a PsiAPI python input file to directory/input.dat
    '''
    if 'call' in kwargs:
        call = kwargs['call'] + '\n\n'
    else:
        call = 'e, wfn = psi4.energy("{}",return_wfn=True)'.format(method)

    if 'directory' in kwargs:
        directory = kwargs['directory']
        try:
            os.makedirs(directory)
        except:
            if os.path.isdir(directory):
                pass
            else:
                raise Exception('Attempt to create {} directory '
                                'failed.'.format(directory))
    else:
        directory = '.'

    if 'module_options' in kwargs:
        module_options = kwargs['module_options']
    else:
        module_options = False

    if 'extra' in kwargs: 
        extra = kwargs['extra']
    else:
        extra = False

    infile = open('{}/input.dat'.format(directory),'w')
    infile.write('# This is a psi4 input file auto-generated for MLQM.\n')
    infile.write('import json\n')
    infile.write('import psi4\n')
    infile.write('psi4.core.set_output_file("output.dat")\n\n')
    infile.write('mol = psi4.geometry("""\n{}\n""")\n\n'.format(molstr))
    infile.write('psi4.set_options(\n{}\n)\n\n'.format(global_options))
    if module_options:
        infile.write('psi4.set_module_options(\n{}\n)\n\n'.format(module_options))
    infile.write('{}\n\n'.format(call))
    infile.write('with open("output.json","w") as dumpf:\n'
                '   json.dump(wfn.variables(), dumpf, indent=4)\n\n')
    if extra:
        infile.write('{}'.format(extra))
    # }}}

def runner(dlist,infile='input.dat'):
    # {{{
    '''
    Run every input in the directory list
    Default input file recommended
    '''
    wd = os.getcwd()
    for d in dlist:
        os.chdir(d)
        subprocess.call(['psi4', 'input.dat', 'output.dat']) 
        os.chdir(wd)
    # }}}

def grabber(dlist,fnames=False,varnames=False,outfile='output.json'):
    # {{{
    '''
    Take in a list of directories and either filenames or variable names
    Filenames should correspond to numpy files to load and return
    Variable names should correspond to variables in a Psi4 JSON output
    Default Psi4 JSON output name is recommended
    Return a result dictionary
    '''
    rdict = {}

    # Harvesting each result separately for now. Worse for IO, but better for 
    # result dictionary structure. 
    if fnames:
        for fname in fnames:
            rdict[fname] = {}
            for dname in dlist:
                rdict[fname][dname] = np.load(dname + '/' + fname)
    if varnames:
        for varname in varnames:
            rdict[varname] = {}
            for dname in dlist:
                with open(dname + '/' + outfile) as out:
                    jout = json.load(out)
                    rdict[varname][dname] = jout[varname]
    return rdict
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

def get_amps(wfn,method):
    # {{{
    '''
    Grab aomplutudes from the wfn
    CCSD just uses wfn.get_amplitudes()
    MP2 builds them from the integrals, Fock, and Hamiltonian
    Returns dictionary with amplitudes
    '''
    if method.upper() == "CCSD":
    # {{{
        # compute and grab amplitudes
        amps = wfn.get_amplitudes()
        t1 = amps['tIA'].to_array()
        t2 = amps['tIjAb'].to_array()
        amps = {'t1':t1,'t2':t2}
    # }}}
    
    elif method.upper() == "MP2":
    # {{{
        # no python access to MP2 amps, compute them by hand
        # see spin-orbital CCSD code in Psi4Numpy
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
        t2 = MOijab / Dijab
        amps = {'t2':t2}
        # }}}

    else:
        print("Automatic amplitude generation from wfn not supported for {}".format(method))
    
    return amps
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

def legacy_pes_gen(ds, **kwargs):
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
                    rep, wfn = repgen.make_tatr(mol,ds.predtype,ds.bas,st=ds.st)
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

