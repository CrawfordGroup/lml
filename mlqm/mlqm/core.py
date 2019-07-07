# Defining a Dataset container class. It will hold the data representations (training,
# validation, testing), the learned model parameters (and hyperparameters), and other
# relevant information.

import numpy as np
import json
from . import train
from . import datahelper

class PES(object):
    """
    A class for holding the information necessary for generating a potential 
    energy surface. Base geometries, displacements, etc. are included here.
    """

    def __init__(self,inp):
    # {{{
        if isinstance(inp,str): # if str, unpack it as a json file
            with open(inp) as f:
                inpf = json.load(f)
            self.inpf = inp
            if 'name' in inpf:
                self.name = inpf['name'] # Name of PES 
            else:
                self.name = 'PES'
            self.geom = inpf['geom'] # Base geometry
            self.gvar = inpf['gvar'] # Variable name in geometry
            self.dis = inpf['dis'] # Displacement range of geometry
            self.pts = inpf['pts'] # Number of points on PES
            self.method = inpf['method']
            self.basis = inpf['basis']
            self.generated = inpf['generated']
            self.complete = inpf['complete']
            print("PES generated from {}.".format(inp))
        elif isinstance(inp,dict): # if a dict, unpack it to class variables
            self.inpf = None
            if 'name' in inp:
                self.name = inp['name'] # Name of PES 
            else:
                self.name = 'PES'
            self.geom = inp['geom'] # Base geometry
            self.gvar = inp['gvar'] # Variable name in geometry
            self.dis = inp['dis'] # Displacement range of geometry
            self.pts = inp['pts'] # Number of points on PES
            self.method = inp['method']
            self.basis = inp['basis']
            self.generated = inp['generated']
            self.complete = inp['complete']
            print("PES generated from input dictionary.")
        else: # initialize an empty database
            self.inpf = None 
            self.name = None
            self.geom = None
            self.gvar = None
            self.dis = None
            self.pts = None
            self.method = None
            self.basis = None
            self.generated = False
            self.complete = False
            print("Empty PES generated. Please supply variables.")
        self.dirlist = None # directory list, may be generated or pushed in later
    # }}}

    def generate(self, global_options=False, regen=False,**kwargs):
        # {{{
        '''
        Generate the input files across a PES.
        Require the method and psi4.set_options dictionary.
        Pass kwargs forward into the input file generation.
        '''
        # generate the grand data set representations
        # N evenly-spaced representations on the PES
        bot = float(self.dis[0])
        top = float(self.dis[1])
        base = self.geom
        gvar = self.gvar
        pes = np.linspace(bot,top,self.pts) 
    
        if not global_options:
            global_options = {
            'basis':self.basis,
            'scf_type':'pk',
            'mp2_type':'conv',
            'freeze_core':'false',
            'e_convergence':1e-8,
            'd_convergence':1e-8}
    
        if 'directory' in kwargs:
            bdir = kwargs['directory']
        else:
            bdir = '.'
                
        if (self.generated == True) and (regen == False):
            dlist = [(bdir + '/{}'.format(dis)) for dis in pes]
            self.dirlist = dlist
            print('PES input files already generated! Pass `regen=True` '
                  ' to regenerate.')
            return dlist
        
        dlist = [] # list of directories
    
        for dis in pes:
            ndir = bdir + '/{}'.format(dis)
            dlist.append(ndir)
            kwargs['directory'] = ndir
            new_geom = base.replace(gvar,str(dis))
            datahelper.write_psi4_input(new_geom,self.method,global_options,**kwargs)

        self.dirlist = dlist
    
        print('PES input files generated! Please run them using the given directory '
              'list. Once complete, you may parse them with pes.parse().')
    
        return dlist
        # }}}

    def run(self,infile='input.dat',restart=False):
        # {{{
        '''
        Run all jobs listed in self.dirlist (may be set manually or by PES.generate())
        NOTE: This simply runs all jobs one-by-one. This is not recommended for large
        datasets, as you are better served running the generated input files
        in parallel. Use at your own risk!
        NOTE: This function could be swapped for a more robust system which keeps 
        track of whether or not jobs have been run, can restart where it left off,
        or other such fancy things. But for now, this is what I'm doing.
        '''
        if self.dirlist == None:
            print('Please pass a directory list into PES.dirlist or have it set '
                  'by running PES.generate().')
            return 0
        elif (self.complete == True) and (restart == False):
            print('PES jobs already complete! Pass `restart=True` to rerun.')
            return 0
        else:
            datahelper.runner(self.dirlist,infile=infile)
            self.complete = True
            return 0
        # }}}

class Dataset(object):
    """
    The main data class for training, validation, and testing representations.
    Setter will initialize a few things, then functions will split and find
    the requested method of learning, validation, etc.
    Maybe I should have a higher class than this, which holds things like the 
    training type, number of trainers (M), etc? Or should I pass the input file
    directly here?
    """

    def __init__(self, inpf=None, loc_str=None):
    # {{{
        """
        Initialize the dataset. Pass in a data set or the location of one.
        """
        # for storing the grand data set from which training sets may be derived
        self.grand = { 
            "representations" : None,
            "values" : None,
            "reference" : None
            }

        if inpf is not None:
            with open(inpf,'r') as f:
                inp = json.load(f)
            self.inpf = inpf
            self.M    = inp['setup']['M']
            self.N    = inp['setup']['N']
            self.st   = inp['setup']['st']
            self.K    = inp['setup']['K']
            self.bas  = inp['setup']['basis']
            self.geom = inp['mol']['geom']
            self.valtype  = inp['setup']['valtype'].upper() # "high" theory for the training set
            self.predtype = inp['setup']['predtype'].upper() # "low" theory for the predictions
            self.ref = inp['setup']['ref'] # generate val set using valtype (for tests)
            print("Using {} amplitudes to predict {}-level energies!".format(self.predtype,self.valtype))
            if (self.valtype == "CCSD-NAT") or (self.predtype == "CCSD-NAT"):
                print("NOTE: Currently, CCSD-NAT can only be used for predicting other CCSD-NAT energies!")

        if loc_str is not None:
            if isinstance(loc_str, str):
                try:
                    self.grand = np.load(loc_str)
                    print("Data loaded from file.")
                except FileNotFoundError:
                    raise RuntimeError("Data could not be loaded from file: {}".format(loc_str))
            elif isinstance(loc_str,np.ndarray):
                self.grand = loc_str
                print("Data loaded.")
            elif isinstance(loc_str,list):
                self.grand = np.asarray(loc_str)
                print("Data loaded.")
            else:
                raise RuntimeError("""Data type {} not recognized. Please provide string 
                        location of numpy file, numpy.ndarray data, or list 
                        data.""".format(type(loc_str)))
        else:
            print("Empty Dataset loaded.")
    # }}}

    def load(self, loc_str):
    # {{{
        if isinstance(loc_str, str):
            try:
                self.grand = np.load(loc_str)
                print("Data loaded from file.")
            except FileNotFoundError:
                raise RuntimeError("Data could not be loaded from file: {}".format(loc_str))
        elif isinstance(loc_str,np.ndarray):
            self.grand = loc_str
            print("Data loaded.")
        elif isinstance(loc_str,list):
            self.grand = np.asarray(loc_str)
            print("Data loaded.")
        else:
            raise RuntimeError("""Data type {} not recognized. Please provide string 
                    location of numpy file, numpy.ndarray data, or list 
                    data.""".format(type(loc_str)))
    # }}}

    def find_trainers(self, traintype, **kwargs):
     # {{{
        if "remove" in kwargs:
            print("NOTE: Trainer map will be valid for grand data set once removed points are dropped!")
        if traintype.lower() in ["kmeans","k-means","k_means"]:
            with open(self.inpf) as f:
                inp = json.load(f)
            if "K" in inp['setup']:
                K = inp['setup']['K']
            else:
                K = 30
            if inp['data']['trainers'] is not False:
                print("{} training set already generated.".format(traintype))
                return inp['data']['trainers']
            else:
                print("Determining training set via {} algorithm . . .".format(traintype))
                trainers = []
                t_map, close_pts = train.k_means_loop(self.grand["representations"],self.M,K,**kwargs)
                for pt in range(0,self.M): # loop over centers, grab positions of trainers
                    trainers.append(t_map[pt][close_pts[pt]][1])
                inp['data']['trainers'] = sorted(trainers, reverse=True)
                with open(self.inpf,'w') as f:
                    json.dump(inp, f, indent=4)
                return sorted(trainers, reverse=True)
        else:
            raise RuntimeError("I don't know how to get {} training points yet!".format(traintype))
    # }}}

    def gen_grand(self, gen_type, **kwargs):
    # {{{
        '''
        DEPRECATION WARNING: moving data generation out of the dataset class!
        '''
        if gen_type.lower() in ["pes"]:
            self.grand["representations"], self.grand["values"], self.grand["reference"] = datahelper.pes_gen(self, **kwargs)
        else:
            print("Generation of {} data not supported.".format(gen_type))
    # }}}

