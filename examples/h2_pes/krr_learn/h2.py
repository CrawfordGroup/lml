import mlqm
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Let's say we have already calculated the relevant energies and made TATR
# representations from MP2 amplitudes
scf_E = np.load('scf_E.npy')
mp2_corr = np.load('mp2_corr.npy')
ccsd_corr = np.load('ccsd_corr.npy')
mp2_reps = np.load('mp2_reps.npy')

# Build Dataset object with settings and data
# Let's try the using MP2 TATRs to predict CCSD energies
ds = mlqm.Dataset(inpf='ml.json', reps=mp2_reps, vals=ccsd_corr)

# Set aside N/4 evenly-spaced points for validation
validators = sorted([i for i in range(0,ds.setup['N'],4)], reverse=True)
valid_reps = [ds.grand["representations"][val] for val in validators]
valid_vals = [ds.grand["values"][val] for val in validators]
ds.grand["representations"] = np.delete(ds.grand["representations"],validators,axis=0)
ds.grand["values"] = np.delete(ds.grand["values"],validators,axis=0)

# Find optimum training set
ds.find_trainers('k-means')
print("Training set: {}".format(ds.data['trainers']))

# Train using kernel ridge regression and save
ds, t_AVG = ds.train('KRR')
ds.save()

# Predict correlation energy across the validation set
pred_E = mlqm.krr.predict(ds,valid_reps)

# Add the SCF energies and average of the training set
pred_E = np.add(pred_E, [scf_E[val] for val in validators])
pred_E = np.add(pred_E, t_AVG)
mp2_E = np.add(mp2_corr,scf_E)
ccsd_E = np.add(ccsd_corr,scf_E)

# Graph the PESs!
pes = np.linspace(0.5,2.0,200)
v_pes = [pes[i] for i in validators]
noval_pes = np.delete(pes,validators,axis=0)
t_pes = [noval_pes[i] for i in ds.data['trainers']]

plt.figure(1,dpi=200)
plt.plot(pes,scf_E,'y-',label='SCF PES')
plt.plot(pes,mp2_E,'r--',label='MP2 PES')
plt.plot(pes,ccsd_E,'b-',linewidth=4,label='CCSD PES')
plt.plot(v_pes,pred_E,'r-^',ms=2,linewidth=2,label='MP2/ML-12 PES')
plt.plot(t_pes,[-1.18 for i in range(0,len(t_pes))],'go',label='Training points')
plt.legend()
plt.savefig('fig.png')
