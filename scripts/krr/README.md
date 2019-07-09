Reproduces the results from [Margraf and Reuter](https://doi.org/10.1021/acs.jpca.8b04455). Algorithm described by the docstring: 

```
The algorithm will displace a diatomic over the range 0.5 - 2 Angstroms, generating
TATRs and corresponding energies at each point. Then, a training set of `M` TATRs
will be selected by the k-means algorithm. Using this training set, the
hyperparameters `s` and `l` (kernel width and regularization) will be determined by
searching over a coarse logarithmic grid and minimizing the regularized L2 squared
loss cost function. Once the hypers are thus determined, the regression coefficients
`a` are trained by solving the corresponding linear equations. A validation set
spread across the PES (not including any training points) is then predicted using the
model, and the results are graphed.
```
