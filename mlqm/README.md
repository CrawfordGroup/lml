# MLQM: Machine-learning quantum mechanics

A python module centered around developing, testing, and deploying machine-learning algorithms to quantum chemical datasets. We have a focus on wave function-based descriptors, with capabilities for building representations from ab initio electronic structure data such as wave function amplitudes or reduced density matrices. 

Beyond representation generation, some `scikit-learn` functions are wrapped for ease-of-use with the `Dataset` class, such as k-means clustering and kernel ridge regression. Defaults are chosen based on performance; however, additional options are available, and `scikit-learn` may also be used directly. See docstrings for details.

To install, run:
```python
pip install -e .
```

To test, run:
```python
py.test
```
from the base `mlqm/` directory.

<!--Currently, kernel-ridge-regression of the CCSD or MP2 amplitudes to CCSD energies is implemented in `krr/krr.py`. Use with `python krr.py data.json` on a clean data file or, if necessary, a partially-filled data file will pick up where it left off. This code is meant to reproduce the [2019 paper](http://dx.doi.org/10.1021/acs.jpca.8b04455) by Margraf and Reuter. The given case is of diatomic hydrogen dissociation (Figure 2 of the paper); however, other diatomics can be easily studied by changing the atoms in the input JSON. See `krr/examples/` for more information. A clean example data file and a filled data file (along with the stored values in `.npy` format) are provided. Graphing modes include `interact` and `save`. Further work will include an interface for going beyond diatomic dissociation curves.--> 
