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

    if 'processors' in kwargs:
        processors = kwargs['processors']
    else:
        processors = False

    if 'extra' in kwargs: 
        extra = kwargs['extra']
    else:
        extra = False

    infile = open('{}/input.dat'.format(directory),'w')
    infile.write('# This is a psi4 input file auto-generated for MLQM.\n')
    infile.write('import json\n')
    infile.write('import psi4\n')
    infile.write('psi4.core.set_output_file("output.dat")\n\n')
    if processors:
        infile.write('psi4.set_num_threads({})\n\n'.format(processors))
    infile.write('mol = psi4.geometry("""\n{}\n""")\n\n'.format(molstr))
    infile.write('psi4.set_options(\n{}\n)\n\n'.format(global_options))
    if module_options:
        infile.write('psi4.set_module_options(\n{}\n)\n\n'.format(module_options))
    infile.write('{}\n\n'.format(call))
    if extra:
        infile.write('{}'.format(extra))
    infile.write('with open("output.json","w") as dumpf:\n'
                '   json.dump(psi4.core.variables(), dumpf, indent=4)\n\n')
    # }}}

def harvest_xyz(dirlist,nlist=None):
    # {{{
    """
    Pass in a list of directories and (optional) accompanying names
    Return a name:geometry dictionary
    Use filename (remove .extension) if no namelist given
    """
    xyz_dict = {}
    if not nlist: # make a namelist from the directories
        nlist = []
        for d in dirlist:
            nlist.append(d.split('.')[0]) # strip the extension
    for d in range(0,len(dirlist)):
        l = 1
        geom = ""
        with open(dirlist[d]) as xyzfile:
            for line in xyzfile:
                if l == 1: # number of atoms
                    l += 1
                    continue
                if l == 2: # comment line
                    l += 1
                    continue
                if l == 3: # xyz start
                    geom += str(line) + "\n"
        xyz_dict[nlist[d]] = geom
    return xyz_dict
    # }}}

def runner(dlist,infile='input.dat',outfile='output.dat'):
    # {{{
    '''
    Run every input in the directory list
    Default input and output files recommended
    '''
    wd = os.getcwd()
    for d in dlist:
        os.chdir(d)
        subprocess.call(['psi4', infile, outfile]) 
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

def get_amps(wfn,method):
    # {{{
    '''
    Grab aomplutudes from the wfn
    CCSD just uses wfn.get_amplitudes()
    MP2 builds them from the integrals, Fock, and Hamiltonian
    NOTE: wfn.get_amplitudes() only works for C1 symmetry, 
    and the MP2 amplitude generation is only intended for C1
    symmetry (AO->MO transformation done w/out regard for
    AO->SO transformation)
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
        # See Psi4NumPy MP2 Gradient code
        # Relevant Variables
        nmo = wfn.nmo()
        nocc = wfn.doccpi()[0]
        nvir = nmo - nocc

        # MO Coefficients
        C = wfn.Ca_subset("AO", "ALL")
        npC = psi4.core.Matrix.to_array(C)

        # Integral generation from Psi4's MintsHelper
        # Build T, V, and S
        mints = psi4.core.MintsHelper(wfn.basisset())
        T = mints.ao_kinetic()
        npT = psi4.core.Matrix.to_array(T)
        V = mints.ao_potential()
        npV = psi4.core.Matrix.to_array(V)
        S = mints.ao_overlap()
        npS = psi4.core.Matrix.to_array(S)

        # Build ERIs
        ERI = mints.mo_eri(C, C, C, C)
        npERI = psi4.core.Matrix.to_array(ERI)
        npERI = npERI.swapaxes(1, 2) # Physicist notation

        # Build Core Hamiltonian and Fock 
        H_core = npT + npV # AO basis
        H = np.einsum('uj,vi,uv', npC, npC, H_core, optimize=True) # MO basis
        F = H + 2.0 * np.einsum('pmqm->pq', npERI[:, :nocc, :, :nocc], optimize=True)
        F -= np.einsum('pmmq->pq', npERI[:, :nocc, :nocc, :], optimize=True)
        F_occ = np.diag(F)[:nocc]
        F_vir = np.diag(F)[nocc:nmo]

        # Build Denominator
        Dijab = F_occ.reshape(-1, 1, 1, 1) + F_occ.reshape(-1, 1, 1) - F_vir.reshape(
            -1, 1) - F_vir

        # Build T2 Amplitudes,
        # where t2 = <ij|ab> / (e_i + e_j - e_a - e_b),
        t2 = npERI[:nocc, :nocc, nocc:, nocc:] / Dijab
        amps = {'t2':t2}
        # }}}

    else:
        print("Automatic amplitude generation from wfn not supported for {}".format(method))
    
    return amps
    # }}}

def harvest_amps(method,namps=150,outfile="output.dat"):
# {{{
    """
    Harvest amplitudes from an output file. Useful for when
    get_amps() cannot be used (ie symmetry not C1). Currently
    only implemented for RHF reference (TIA and TIjAb amps)
    """
    if method.upper() == "MP2":
        # MP2 amps print before CCSD amps
        mcheck = ["Largest TIjAb Amplitudes:"]
        mend = ["Solving CC Amplitude Equations"]
        ttype = ['t2']
        amps = {'t2':[]}
    elif method.upper() == "CCSD":
        mcheck = ["Largest TIA Amplitudes:","Largest TIjAb Amplitudes:"]
        mend = ["Largest TIjAb Amplitudes:","SCF energy"]
        ttype = ['t1','t2']
        amps = {'t1':[],'t2':[]}
    else:
        raise Exception("Cannot harvest {} amplitudes.".format(method))

    get_line = 0
    get_next = 0
    t = 0 # amplitudes will always be printed in order t1, t2, t3 . . .
    with open(outfile) as f:
        for line in f:
            if (t+1 > len(ttype)): # harvested all ttypes
                break
            elif mcheck[t] in line: # found method check, start collecting amps
                get_next = 1
            elif (get_next == 1) and (get_line < namps): # passed mcheck, need more amps
                if mend[t] in line: # stop collecting amps
                    get_next = 0
                    get_line = 0
                    t += 1
                    continue
                try: # try to collect the amp
                    if ttype[t] == 't1':
                        _,_,amp = line.split() # I A amp
                    elif ttype[t] == 't2':
                        _,_,_,_,amp = line.split() # I j A b amp
                    else:
                        raise Exception("Can't handle {} amps just yet!".format(ttype[t]))
                    amps[ttype[t]].append(float(amp))
                    get_line += 1
                except: # probably an empty line
                    pass
            elif get_line >= namps: # got all of the requested amps
                get_next = 0
                t += 1
            else: # haven't hit mcheck yet
                pass

#            if (get_next == 1) & (get_line < namps):
#                if (t < len(ttype) - 1):
#                    if mcheck[t+1] in line: # just in case amps run out before namps
#                        get_line = 0
#                        t += 1
#                        pass
#                try:
#                    if ttype[t] == 't1':
#                        _,_,amp = line.split()
#                    elif ttype[t] == 't2':
#                        _,_,_,_,amp = line.split()
#                    else:
#                        raise Exception("Can't handle {} amps just yet!".format(ttype[t]))
#                    amps[ttype[t]].append(float(amp))
#                    get_line += 1
#                except:
#                    pass
#            elif mcheck[t] in line:
#                get_next += 1
#            elif (get_line >= namps) and (t < len(ttype) - 1):
#                get_line = 0
#                get_next = 0
#                t += 1
#            else:
#                pass

    for t in amps:
        amps[t] = np.asarray(amps[t])
    return amps
# }}}

def reg_l2(y,y_p,l,a):
# {{{
    '''
    Calculate L2 (squared) loss with regularization to protect against
    large norm squared regression coefficients
    given true and predicted values, regularization, and regression coefficients
    '''
#    ls = sum([(y_p[i] - y[i])**2 for i in range(len(y))]) / len(y)
    ls = np.linalg.norm(y-y_p)**2 + l*np.linalg.norm(a)**2
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

