# Defining a Dataset container class. It will hold the data representations (training,
# validation, testing), the learned model parameters (and hyperparameters), and other
# relevant information.

import numpy as np
import json
import copy
from . import train
from . import datahelper
from . import krr
import os

class Mol_Set(object):
    """
    A class for holding the information necessary to create a molecular
    dataset. Run Mol_Set.generate() to generate input files across the set, 
    and Mol_Set.run() to automatically go through the directories and run 
    each input file. Check the respective fn docstrings to see extra 
    capabilities.
    """

    def __init__(self,inp=None,info=None):
    # {{{
        """
        Initialize the molecular dataset. 
        inp is an input dict with '[molname]':'[geometry]' 
        If None (default), empty Mol_Set initialized
        info is a docstring about the dataset
        """
        #TODO: add an option for passing in a list of directories
        if isinstance(inp,dict):
#            self.geometries = list(inp.values())
#            self.names = list(inp.keys())
            self.geometries = copy.deepcopy(inp)
            self.size = len(inp)
            self.info = info
            self.generated = False 
            self.complete = False 
            print("Molecule dataset generated from input dictionary.")
        else: # initialize an empty database
            self.geometries = list(inp.values())
            self.names = list(inp.keys())
            self.size = len(inp)
            self.info = info
            self.generated = False 
            self.complete = False 
            print("Empty molecule dataset generated. Please supply variables.")
        self.dirlist = None # to be set with generator
    # }}}

    def save(self, loc=False):
        pass

    def generate(self, method, global_options=False, regen=False, **kwargs):
        # {{{
        '''
        Generate the input files across a Mol_Set.
        Require the method. 
        Optional psi4.set_options dictionary.
        Pass kwargs forward into the input file generation.
        '''
    
        if not global_options:
            global_options = {
            'basis':'sto-3g',
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
            dlist = [(bdir + '/{}'.format(name)) for name in list(self.geometries.keys())]
            self.dirlist = dlist
            print('Dataset input files already generated! Pass `regen=True` '
                  'to regenerate.')
            return dlist
        
        dlist = [] # list of directories
        for n,g in self.geometries.items():
            ndir = bdir + '/{}'.format(n)
            dlist.append(ndir)
            kwargs['directory'] = ndir
            geom = copy.deepcopy(g) 
            datahelper.write_psi4_input(geom,method,global_options,**kwargs)

        self.dirlist = dlist
        self.generated = True
    
        print('Dataset input files generated! Please run them using the given directory '
              'list. Once complete, you may parse them with dataheler.grabber().')
    
        return dlist
        # }}}

    def run(self,infile='input.dat',outfile='output.dat',restart=False, **kwargs):
        # {{{
        '''
        Run all jobs listed in self.dirlist (may be set manually or by Mol_Set.generate())
        NOTE: This simply runs all jobs one-by-one. This is not recommended for large
        datasets, as you are better served running the generated input files
        in parallel. Use at your own risk!
        '''
        if self.dirlist == None:
            print('Please pass a directory list into Mol_Set.dirlist or have it set '
                  'by running Mol_Set.generate().')
            return 0
        elif (self.complete == True) and (restart == False):
            print('Mol_Set jobs already complete! Pass `restart=True` to rerun.')
            return 0
        else:
            datahelper.runner(self.dirlist,infile=infile,outfile=outfile, **kwargs)
            self.complete = True
            return 0
        # }}}

class PES(object):
    """
    A class for holding the information necessary for generating a potential 
    energy surface. Base geometries, displacements, etc. are included here.
    Run pes.generate() to generate input files across the PES, and pes.run()
    to automatically go through the directories and run each input file.
    Check the respective fn docstrings to see extra capabilities.

    Keyword Args:
    raise_on_error = True :
        Whether to raise an error when the input file is not found (True),
        or to make an empty PES if it is not found (False).
    """

    def __init__(self,inp, **kwargs):
    # {{{
        if isinstance(inp,str): # if str, unpack it as a json file
            if not os.path.isfile(inp) and \
               (("raise_on_error" in kwargs and kwargs["raise_on_error"])
                or "raise_on_error" not in kwargs) :
                raise FileNotFoundError(f"Could not find the file {inp}:"
                                        + " Can not create PES")
            elif not os.path.isfile(inp) and ("raise_on_error" in kwargs and
                                              not kwargs["raise_on_error"]) :
                print(f"Could not find the file {inp}: Initializing empty PES")
                self.geom = None
                self.gvar = None
                self.dis = None
                self.pts = None
                self.method = None
                self.basis = None
                self.generated = None
                self.complete = None
                self.inpf = inp
                return
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

    def save(self, loc=False):
        # {{{
        if self.inpf:
            loc_str = self.inpf
        else:
            loc_str = loc
        settings = {
           "inpf": loc_str,
           "name": self.name,
           "geom": self.geom,
           "method": self.method,
           "pts": self.pts,
           "gvar": self.gvar,
           "dis": self.dis,
           "basis": self.basis,
           "generated": self.generated,
           "complete": self.complete 
        }
        with open(loc_str,'w') as f:
            json.dump(settings,f,indent=4)
        # }}}

    def generate(self, global_options=False, pestype='INCLUDE', regen=False,**kwargs):
        # {{{
        '''
        Generate the input files across a PES.
        Require the method and psi4.set_options dictionary.
        Pass kwargs forward into the input file generation.
        Note that datahelper.grabber() can get any variable from a Psi4 output 
        JSON, as well as load data from any arbitrarily-named NumPy file!
        Example: if you pass `extra = "np.save('amps.npy',t1_amps)`,
        this will be placed at the bottom of every input file.
        Then, you can run grabber(dirlist,fnames=['amps.npy']) and the
        returned result dict will include your amplitudes!
        '''
        # generate the grand data set representations
        # 'pts' evenly-spaced representations on the PES
        bot = float(self.dis[0])
        top = float(self.dis[1])
        base = self.geom
        gvar = self.gvar
        if pestype.upper() == "INCLUDE":
            pes = np.linspace(bot,top,self.pts) 
        elif pestype.upper() == "EXCLUDE":
            pes = [bot]
            x = bot
            for i in range(0,self.pts-1):
                x += (top - bot) / self.pts
                pes.append(x)
            pes = np.round(pes,4)
        else:
            raise Exception("PES type {} not recognized.".format(pestype))
    
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
        self.generated = True
    
        print('PES input files generated! Please run them using the given directory '
              'list. Once complete, you may parse them with dataheler.grabber().')
    
        return dlist
        # }}}

    def run(self,infile='input.dat',outfile='output.dat',restart=False, **kwargs):
        # {{{
        '''
        Run all jobs listed in self.dirlist (may be set manually or by PES.generate())
        NOTE: This simply runs all jobs one-by-one. This is not recommended for large
        datasets, as you are better served running the generated input files
        in parallel. Use at your own risk!
        '''
        if self.dirlist == None:
            print('Please pass a directory list into PES.dirlist or have it set '
                  'by running PES.generate().')
            return 0
        elif (self.complete == True) and (restart == False):
            print('PES jobs already complete! Pass `restart=True` to rerun.')
            return 0
        else:
            datahelper.runner(self.dirlist,infile=infile,outfile=outfile, **kwargs)
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

    Keyword Args:
    raise_on_error = True :
        Whether to raise an error when the input file is not found (True),
        or to make an empty Dataset if it is not found (False).
    """

    def __init__(self, inpf=None, reps=None, vals=None, **kwargs):
    # {{{
        """
        Initialize the dataset. Pass in a data set or the location of one.
        """
        # for storing the grand data set representations and values 
        self.grand = { 
            "representations" : None,
            "values" : None,
            }

        if inpf is not None:
            if isinstance(inpf,str):
                if not os.path.isfile(inpf) and \
                   (("raise_on_error" in kwargs and kwargs["raise_on_error"]) or
                     "raise_on_error" not in kwargs) :
                    raise FileNotFoundError(f"Could not find the file {inpf}:"
                                        + " Can not create Dataset.")
                elif not os.path.isfile(inpf) and \
                    ("raise_on_error" in kwargs and not kwargs["raise_on_error"]):
                     print(f"Could not find the file {inpf}: Initializing empty"
                           + " Dataset")
                     self.inpf = inpf
                     self.setup = None
                     self.data = None
                with open(inpf,'r') as f:
                    inp = json.load(f)
                self.inpf = inpf
                self.setup = inp['setup']
                self.data = inp['data']
            elif isinstance(inpf,dict):
                self.inpf = None
                self.setup = inpf['setup']
                self.data = inpf['data']
            else:
                print("Please pass in either a STR json filepath or DICT.")
        else:
            self.setup = None
            self.data = None

        if reps is not None:
            if isinstance(reps,str):
                self.grand["representations"] = np.load(reps)
            elif isinstance(reps,np.ndarray):
                self.grand["representations"] = reps
            elif isinstance(reps,list):
                self.grand["representations"] = np.asarray(reps)
            else:
                raise RuntimeError("""Please pass in a STR numpy filepath, numpy.ndarray, or 
                        list of representations.""")

        if vals is not None:
            if isinstance(vals,str):
                self.grand["values"] = np.load(vals)
            elif isinstance(vals,np.ndarray):
                self.grand["values"] = vals
            elif isinstance(vals,list):
                self.grand["values"] = np.asarray(vals)
            else:
                raise RuntimeError("""Please pass in a STR numpy filepath, numpy.ndarray, or 
                        list of values.""")
        try:
            if (inpf == None) and (reps == None) and (vals == None):
                self.inpf = None
                self.setup = {}
                self.data = {}
                print("Empty Dataset loaded.")
        except:
            print("Partial Dataset loaded.")
    # }}}

    def load(self, inpf=None, reps=None, vals=None):
        # {{{
        '''
        Pass in either a STR json filepath or DICT for setup, data, and grand.
        '''
        if inpf is not None:
            if isinstance(inpf,str):
                with open(inpf,'r') as f:
                    inp = json.load(f)
                self.inpf = inpf
                self.setup = inp['setup']
                self.data = inp['data']
            elif isinstance(inpf,dict):
                self.inpf = None
                self.setup = inpf['setup']
                self.data = inpf['data']
            else:
                print("Please pass in either a STR json filepath or DICT.")

        if reps is not None:
            if isinstance(reps,str):
                self.grand["representations"] = np.load(reps)
            elif isinstance(reps,np.ndarray):
                self.grand["representations"] = reps
            elif isinstance(reps,list):
                self.grand["representations"] = np.asarray(reps)
            else:
                raise RuntimeError("""Please pass in a STR numpy filepath, numpy.ndarray, or 
                        list of representations.""")

        if vals is not None:
            if isinstance(vals,str):
                self.grand["values"] = np.load(vals)
            elif isinstance(vals,np.ndarray):
                self.grand["values"] = vals
            elif isinstance(vals,list):
                self.grand["values"] = np.asarray(vals)
            else:
                raise RuntimeError("""Please pass in a STR numpy filepath, numpy.ndarray, or 
                        list of values.""")
        # }}}

    def save(self, loc=False):
        # {{{
        if self.inpf:
            loc_str = self.inpf
        else:
            loc_str = loc
        with open(loc_str,'w') as f:
            json.dump({'setup':self.setup,'data':self.data},f,indent=4)
        # }}}

    def find_trainers(self, traintype, **kwargs):
    # {{{
        '''
        Passes the dataset and any kwargs into the appropriate training set optimization.
        '''
        if "remove" in kwargs:
            print("NOTE: Trainer map will be valid for grand data set once removed points are dropped!")
        if "K" in self.setup:
            K = self.setup['K']
        else:
            K = 30

        if "trainers" in self.data:
            if self.data['trainers'] is not False:
                print("{} training set already generated.".format(traintype))
                return self.data['trainers']
            else:
                pass
        self.data['trainers'] = False

        if traintype.lower() in ["kmeans","k-means","k_means"]:
            # {{{
            print("Determining training set via {} algorithm . . .".format(traintype))
#            trainers = []
#            t_map, close_pts = train.k_means_loop(self.grand["representations"],self.setup['M'],K,**kwargs)
#            for pt in range(0,self.setup['M']): # loop over centers, grab positions of trainers
#                trainers.append(t_map[pt][close_pts[pt]][1])
            from sklearn.cluster import KMeans
            from sklearn.metrics import pairwise_distances_argmin_min
            if 'remove' in kwargs:
                print("Removing given points from validation set.")
                pts = np.delete(self.grand["representations"],kwargs['remove'],axis=0)
            else:
                pts = self.grand["representations"]
            kmeans = KMeans(n_clusters=self.setup['M'],n_init=K,tol=0).fit(pts)
            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, pts)
            trainers = sorted(closest,reverse=True)
            trainers = [int(t) for t in trainers] # set to INT so JSON serializable
            self.data['trainers'] = sorted(trainers, reverse=True)
            return sorted(trainers, reverse=True)
            # }}}

        if traintype.lower() in ["home_kmeans","home_k-means","home_k_means"]:
            # {{{
            print("Determining training set via {} algorithm . . .".format(traintype))
            print("NOTE: This is a home-baked k-means routine. User discretion is advised.")
            trainers = []
            t_map, close_pts = train.k_means_loop(self.grand["representations"],self.setup['M'],K,**kwargs)
            for pt in range(0,self.setup['M']): # loop over centers, grab positions of trainers
                trainers.append(t_map[pt][close_pts[pt]][1])
            self.data['trainers'] = sorted(trainers, reverse=True)
            return sorted(trainers, reverse=True)
            # }}}

        else:
            raise RuntimeError("I don't know how to get {} training points yet!".format(traintype))
    # }}}

    def train(self, traintype, **kwargs):
        # {{{
        '''
        Passes the dataset and any kwargs into the appropriate training set optimization.
        '''
        if 'loss' in kwargs:
            loss = kwargs['loss']
        else:
            loss = 'neg_mean_squared_error' # erratic for LiF
#            loss = 'neg_mean_absolute_error' # erratic
#            loss = 'explained_variance' # smooth but wrong for LiF
#            loss = 'neg_median_absolute_error' # erratic
            # skl default is 'r2' but it's terrible
        if 'kernel' in kwargs:
            kernel = kwargs['kernel']
        else:
            kernel = 'rbf'

        if traintype.lower() in ["krr","kernel_ridge"]:
            # {{{
            print("Training via the {} algorithm . . .".format(traintype))
#            ds, t_AVG = krr.train(self, **kwargs) 
            from sklearn.kernel_ridge import KernelRidge
            from sklearn.model_selection import GridSearchCV

            t_REPS = [self.grand['representations'][tr] for tr in self.data['trainers']]
            t_VALS = [self.grand['values'][tr] for tr in self.data['trainers']]
            t_AVG = np.mean(t_VALS)
            t_VALS = np.subtract(t_VALS,t_AVG)

            # get the hypers s(igma) = kernel width and l(ambda) = regularization
            # NOTE: while my input file uses "s" and "l", skl treats these
            # as "gamma" and "alpha" where gamma = 1/(2*s**2) and alpha = l
            # TODO: Depending on the kernel, s/gamma may not be necessary
            if 'hypers' in self.data and self.data['hypers']:
                print("Loading hyperparameters from Dataset.")
                gamma = 1.0 / 2.0 / self.data['s']**2 
                alpha = self.data['l'] 
                krr = KernelRidge(kernel=kernel,alpha=alpha,gamma=gamma)
            else:
                krr = KernelRidge(kernel=kernel)
                if 'k' in kwargs:
                    k = kwargs['k']
                else:
                    k = self.setup['M']
                parameters = {'alpha':np.logspace(-12,12,num=50),
                              'gamma':np.logspace(-12,12,num=50)}
                krr_regressor = GridSearchCV(krr,parameters,scoring=loss,cv=k)
                krr_regressor.fit(t_REPS,t_VALS)
                self.data['hypers'] = True
                self.data['s'] = 1.0 / (2*krr_regressor.best_params_['gamma'])**0.5
                self.data['l'] = krr_regressor.best_params_['alpha']
                krr = krr_regressor.best_estimator_
        
            # train for a(lpha) = regression coefficients
            # NOTE: I call the coeffs "a" while skl uses "dual_coef"
            if ('a' in self.data) and self.data['a']:
                print("Loading coefficients from Dataset.")
                alpha = np.asarray(self.data['a'])
                krr.dual_coef_ = alpha
            else:
                print("Model training using s = {} and l = {} . . .".format(self.data['s'],self.data['l']))
                krr.fit(t_REPS,t_VALS)
                self.data['a'] = list(krr.dual_coef_)
            return self, t_AVG
            # }}}

        elif traintype.lower() in ["home_krr","home_kernel_ridge"]:
            print("Training via the {} algorithm . . .".format(traintype))
            print("NOTE: This algorithm only includes the radial basis function with a custom loss function.")
            ds, t_AVG = krr.train(self, **kwargs) 
            return ds, t_AVG

        else:
            raise RuntimeError("Cannot train using {} yet.".format(traintype))
        # }}}
