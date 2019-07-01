import mlqm
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

a = mlqm.core.Dataset('trial.json')

a.gen_grand('pes')

# N/4 evenly spaced validation points
validators = sorted([i for i in range(0,a.N,4)], reverse=True) 

trainers = a.find_trainers("k-means", remove=validators)

v_SCF, pred_E_list, v_E_list = mlqm.krr.krr(a,trainers,validators,remove=True)

pes = np.linspace(0.5,2.0,200)
v_PES = [pes[i] for i in validators]
noval_pes = np.delete(pes,validators,axis=0)
t_pos = [noval_pes[i] for i in trainers]

plt.figure(1,dpi=100)
plt.plot(v_PES,v_SCF,'y-',label='SCF')
plt.plot(v_PES,v_E_list,'b-o',label='{} PES'.format('CCSD'),linewidth=3)
plt.plot(v_PES,pred_E_list,'r-^',ms=2,label='{}/ML-{}'.format('MP2',a.M),linewidth=2)
plt.plot(t_pos,[-1.18 for i in range(0,len(t_pos))],'go',label='Training points')
plt.axis([0.5,2.0,-1.2,-0.9])
plt.xlabel('r/Angstrom')
plt.ylabel('Energy/$E_h$')
plt.legend()
plt.show()
