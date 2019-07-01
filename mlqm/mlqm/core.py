# Defining a Dataset container class. It will hold the data representations (training,
# validation, testing), the learned model parameters (and hyperparameters), and other
# relevant information.

import numpy as np
import json
from . import train
from . import datahelper

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
            self.ref = inp['setup']['ref'] # store and graph validation set with val and predtype
            print("Using {} amplitudes to predict {}-level energies!".format(self.predtype,self.valtype))

        if loc_str is not None:
            if isinstance(loc_str, str):
                try:
                    self.grand = np.load(loc_str).to_list()
                    print("Data loaded from file.")
                except FileNotFoundError:
                    raise RuntimeError("Data could not be loaded from file: {}".format(loc_str))
            elif isinstance(loc_str,np.ndarray):
                self.data = loc_str
                print("Data loaded.")
            elif isinstance(loc_str,list):
                self.data = np.asarray(loc_str)
                print("Data loaded.")
            else:
                raise RuntimeError("""Data type {} not recognized. Please provide string 
                        location of numpy file, numpy.ndarray data, or list 
                        data.""".format(type(loc_str)))
        else:
            print("Empty Dataset loaded.")

    def load(self, loc_str):
        if isinstance(loc_str, str):
            try:
                self.grand = np.load(loc_str).to_list()
                print("Data loaded from file.")
            except FileNotFoundError:
                raise RuntimeError("Data could not be loaded from file: {}".format(loc_str))
        elif isinstance(loc_str,np.ndarray):
            self.data = loc_str
            print("Data loaded.")

    def gen_grand(self, gen_type, **kwargs):
        if gen_type.lower() in ["pes"]:
            self.grand["representations"], self.grand["values"], self.grand["reference"] = datahelper.pes_gen(self, **kwargs)
        else:
            print("Generation of {} data not supported.".format(gen_type))

    def find_trainers(self, traintype, **kwargs):
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

    def gen_train(self, train_type, **kwargs):
        """
        Push the data into the appropriate generation routine for the trainers.

        Examples
        --------
        data.gen_train('k_means', K=12)
        """

        if train_type.lower() in ["k_means","kmeans","k-means"]:
            return train.k_means(**kwargs)

