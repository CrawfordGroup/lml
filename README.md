# LML: Learning Machine Learning

Here lies a collection of tutorials and scripts compiled during the course of learning about machine learning algorithms, particularly as they pertain to chemical applications. Generally `numpy` is necessary, and `scikit-learn` may also be used in some places. Machine-learning quantum mechanics (MLQM) scripts will generally use [`Psi4`](https://github.com/psi4/psi4) as a driver for generating test and validation data. (NOTE: [`mesp`](https://github.com/bgpeyton/mesp) may also be used during development of MLQM algorithms)

## Tutorials

These are the first steps towards applying machine-learning algorithms to problems in a straightforward, visual way. These tutorials are built in Jupyter notebooks, and are highly inspired by the [`Psi4NumPy`](https://github.com/psi4/psi4numpy) repository. Static versions of the tutorials can be viewed on GitHub; however, it is suggested the user downloads them and experiments with them to truly understand the power of some of these simple methods.

<img src="https://github.com/bgpeyton/lml/blob/master/images/MLR_jup.png" height=250>

## Scripts

Scripts which are neither tutorials nor full MLQM applications will go here. These may be generalizations of the above tutorials or simply tests which are worth keeping around for future work or instructive purposes. Generally they come with tests, which provide a sample application of the algorithm.

<img src="https://github.com/bgpeyton/lml/blob/master/images/kmeans.png" height=250>

## MLQM

The main purpose of this repository is the development of the building blocks for machine-learning quantum mechanics algorithms. As such, the tutorials and scripts mentioned above generally pertain to approaches that are or will be seen in `/mlqm/`. These algorithms may or may not be fully realized into novel approaches to predicting quantum chemical data-- if they are merely for the recreation or re-purposing of literature results, references will be given. 

Fully implemented algorithms will be added to the MLQM Package, which can be installed by running: 
```python
pip install -e .
```
inside the `/lml/mlqm/` directory (where `setup.py` is located). The purpose of this package is to facilitate the mixing of different algorithms and quick testing of alternative data types and intermediate steps during the learning process. 

The package is based around the `Dataset` class for machine-learning applications, but also has a `PES` (Potential Energy Surface) class to facilitate the generation of potential energy surface data to use for testing MLQM algorithms. `datahelper.py` also allows for quick generation, running, and result-parsing of Psi4 calculations (including adding and parsing additional arbitrary data saved as .npy files, see `/examples/h2_pes/input_gen/` for more details). However, the code is meant to be used on more than just PES data, and so the `Dataset` class can be populated with data from arbitrary sources (see `/examples/h2_pes/krr_learn/` for more details).

<img src="https://github.com/bgpeyton/lml/blob/master/images/krr.png" height=350>
