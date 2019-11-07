#!/usr/bin/python3

from . import base

import psi4
import numpy as np
import json
import os

class Psi4InputGenerator(base.InputGenerator) :
    """
    Psi4InputGenerator

    This class builds Psi4 input files from a set of molecules.
    """

    __singleton = None
    
    def __init__(self) :
        pass

    def getsingleton() :
        if Psi4InputGenerator.__singleton is None :
            Psi4InputGenerator.__singleton = Psi4InputGenerator()
        return Psi4InputGenerator.__singleton

    def write_psi4_input(molstr,method,global_options,**kwargs):
    # {{{
        '''
        Pass in a molecule string, method, and global options dictionary
        Kwargs to hold optional directory, module options dictionary,
        alternative calls (ie properties) and extra commands as strings 
        Writes a PsiAPI python input file to directory/input.dat
        '''
        if 'geomfunc' in kwargs:
            molstr += '\n' + kwargs['geomfunc']

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

        if 'memory' in kwargs:
            memory = kwargs['memory']
        else:
            memory = False

        if 'extra' in kwargs: 
            extra = kwargs['extra']
        else:
            extra = False

        infile = open('{}/input.dat'.format(directory),'w')
        infile.write('# This is a psi4 input file auto-generated for MLQM.\n')
        infile.write('import json\n')
        infile.write('import psi4\n')
        infile.write(f'psi4.core.set_output_file("{directory}/output.dat")\n\n')
        if processors:
            infile.write('psi4.set_num_threads({})\n'.format(processors))
        if memory:
            infile.write('psi4.set_memory("{}")\n'.format(memory))
        infile.write('mol = psi4.geometry("""\n{}\n""")\n\n'.format(molstr))
        infile.write('psi4.set_options(\n{}\n)\n\n'.format(global_options))
        if module_options:
            infile.write('psi4.set_module_options(\n{}\n)\n\n'.format(module_options))
        infile.write('{}\n\n'.format(call))
        if extra:
            infile.write('{}'.format(extra))
        infile.write(f'with open("{directory}/output.json","w") as dumpf:\n'
                    '   json.dump(psi4.core.variables(), dumpf, indent=4)\n\n')
        # }}}
        
    def mol_to_psi4(molecule) :
        geoms = molecule.geometry()
        out = f'{molecule.charge()} {molecule.spinstate()}' + \
              ''.join(f'\n{geoms[i][0]}@{geoms[i][1]} {geoms[i][2]} {geoms[i][3]} {geoms[i][4]}'
                      for i in range(len(geoms)))
        return out

    def generate(self, molset, directory, method, opts, **kwargs) :
        """
        Generate the input files and directories for Psi4. If
        'include_meta' is passed and true, then the metadata for each molecule
        will be written to a file called 'meta.npy'.

        Returns
        -------
        iterable[str]
            A list of directories containing Psi4 input files.
        """
        out = []
        if not os.path.isdir(directory) :
            os.mkdir(directory)
        
        for mol in molset :
            ndir = directory + f'/{str(mol)}'
            out.append(ndir)
            if os.path.isfile(ndir + "/input.dat") and ("regen" not in
                                                        kwargs or not
                                                        kwargs["regen"]) :
                continue
            Psi4InputGenerator.write_psi4_input(
                Psi4InputGenerator.mol_to_psi4(mol), method, opts,
                directory = ndir, **kwargs)
            if 'include_meta' in kwargs and kwargs['include_meta'] :
                np.save(ndir + "/meta.npy", mol.metadata())
        return out

            
