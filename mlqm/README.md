# MLQM: Machine-learning quantum mechanics

This is where I store short or in-progress machine learning algorithms that pertain to quantum mechanics. Eventually, MLQM may get its own repository, or the codes housed within will get their own separate repositories. 

Currently, kernel-ridge-regression of the CCSD amplitudes to CCSD energies is implemented in `krr.py`. This code is meant to reproduce the [2019 paper](http://dx.doi.org/10.1021/acs.jpca.8b04455) by Margraf and Reuter. The given case is of diatomic hydrogen dissociation (Figure 2 of the paper); however, other diatomics can be easily studied by changing the atoms in the total data set generation step. Further work will include the option of using MP2 amplitudes for the validation set, and a cleaner interface for going beyond diatomic dissociation curves. 
