# LML: Learning Machine Learning

Here lies a collection of tutorials and scripts compiled during the course of learning about machine learning algorithms, particularly as they pertain to chemical applications. Generally `numpy` is necessary, and `scikit-learn` may also be used in some places. Machine-learning quantum mechanics (MLQM) scripts will generally use [`Psi4`](https://github.com/psi4/psi4) as a driver for generating test and validation data. (NOTE: [`mesp`](https://github.com/bgpeyton/mesp) may also be used during development of MLQM algorithms)

## Tutorials

These are the first steps towards applying machine-learning algorithms to problems in a straightforward, visual way. These tutorials are built in Jupyter notebooks, and are highly inspired by the [`Psi4NumPy`](https://github.com/psi4/psi4numpy) repository. Static versions of the tutorials can be viewed on GitHub; however, it is suggested the user downloads them and experiments with them to truly understand the power of some of these simple methods.

<img src="https://github.com/bgpeyton/lml/blob/master/images/MLR_jup.png" height=250>

## Scripts

Scripts which are neither tutorials nor full MLQM applications will go here. These may be generalizations of the above tutorials or simply tests which are worth keeping around for future work or instructive purposes. Generally they come with tests, which provide a number of sample applications of the algorithm.

<img src="https://github.com/bgpeyton/lml/blob/master/images/kmeans.png" height=250>

## MLQM

The main purpose of this repository is the development of the building blocks for machine-learning quantum mechanics algorithms. As such, the tutorials and scripts mentioned above generally pertain to approaches that are or will be seen in `/mlqm/`. These algorithms may or may not be fully realized into novel approaches to predicting quantum chemical data-- if they are merely for the recreation or re-purposing of literature results, references will be given. 

Some will also be collected into the MLQM Package, which can be installed by running: 
```python
pip install -e .
```
The purpose of this package is to facilitate the mixing of different algorithms, and also quick testing of alternative data types and intermediate steps during the learning process. 

<img src="https://github.com/bgpeyton/lml/blob/master/images/krr.png" height=350>
