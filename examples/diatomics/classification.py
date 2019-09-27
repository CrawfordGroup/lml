#!/usr/bin/python3

import mlqm
import math
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import psi4
import matplotlib.colors as colors

extra_stuff = """
wfn.to_file('wfn.npy')
import mlqm
import numpy as np
mp2_amps = mlqm.datahelper.get_amps(wfn, 'MP2')
np.save('mp2_amps.npy', mp2_amps['t2'])
"""

if not os.path.isdir("images/classification") :
    os.mkdir("images/classification")

def colormap(i, n):
    return colors.to_hex(colors.hsv_to_rgb([i / n, 1, 1]))

scf_Es = [] #SCF energies
mp2_biggest_reps = []   #Biggest MP2 rep
mp2_Es = [] #MP2 energies
molecule = []   #Which molecule each point belongs to
names = []
masses = []

fd = open('files.json')
sets = json.load(fd)
fd.close()

n = 0
for f in sets['members'] :
    pes = mlqm.PES('%s/input.json'%(s))

    if pes.complete and \
       all(os.path.isfile(d) for d in ['%s/pes/%s.npy'%(pes.name, n)
                                       for n in ['scf_E', 'mp2_corr',
                                                 'ccsd_corr', 'mp2_reps']]) \
                            and os.path.isfile(f'{pes.name}/pes/mass.npy'):
        scf_E = np.load('%s/pes/scf_E.npy'%(pes.name))
        mp2_corr = np.load('%s/pes/mp2_corr.npy'%(pes.name))
        ccsd_corr = np.load('%s/pes/ccsd_corr.npy'%(pes.name))
        mp2_reps = np.load('%s/pes/mp2_reps.npy'%(pes.name))
        mass = np.load('%s/pes/mass.npy'%(pes.name))
        
    else :   
        dlist = pes.generate({'basis': pes.basis, 'scf_type': 'pk',
                                 'mp2_type': 'conv', 'freeze_core': 'false',
                                 'e_convergence':1e-8, 'd_convergence':1e-8},
                                directory='%s/pes'%(pes.name),
                                extra = extra_stuff)
        pes.save()
        pes.run(progress=True)
        pes.save()
        results = mlqm.datahelper.grabber(dlist, varnames=[
            'MP2 CORRELATION ENERGY', 'HF TOTAL ENERGY',
            'CCSD CORRELATION ENERGY'],
                                             fnames = ['mp2_amps.npy'])
        scf_E = list(results['HF TOTAL ENERGY'].values())
        mp2_corr = list(results['MP2 CORRELATION ENERGY'].values())
        ccsd_corr = list(results['CCSD CORRELATION ENERGY'].values())

        mp2_reps = [mlqm.repgen.make_tatr('MP2', results['mp2_amps.npy'][d])
                       for d in dlist]
        wfn = psi4.core.Wavefunction.from_file(
            '%s/wfn.npy'%(list(results["mp2_amps.npy"].keys())[0]))
        mol = wfn.molecule()
        mass = sum(mol.mass(i) for i in range(mol.natom()))
        print(mass)
        

        np.save('%s/pes/scf_E.npy'%(pes.name), scf_E)
        np.save('%s/pes/mp2_corr.npy'%(pes.name), mp2_corr)
        np.save('%s/pes/ccsd_corr.npy'%(pes.name), ccsd_corr)
        np.save('%s/pes/mp2_reps.npy'%(pes.name), mp2_reps)
        np.save('%s/pes/mass.npy'%(pes.name), mass)

    scf_Es = scf_Es + list([x / mass for x in scf_E])
    mp2_Es += list(mp2_corr)
    mp2_biggest_reps += [max(mp2_reps[i], key = lambda a: abs(a))
                         for i in range(len(mp2_reps))]
    masses += [mass]
    molecule += [n for i in range(len(scf_E))]
    names.append(pes.name)
    n += 1

#Make pictures of each plot.

#scf v mp2_reps
#without groups
plt.figure(dpi=200)
plt.scatter(scf_Es, mp2_biggest_reps, 0.5)
plt.title("SCF energy vs. biggest (absolute) MP2 rep")
plt.xlabel("SCF energy")
plt.ylabel("MP2 rep value")
plt.savefig("images/classification/scf_reps.png")

#with groups
plt.figure(dpi=200)
for i in range(n) :
    plt.scatter([scf_Es[j] for j in range(len(scf_Es)) if i == molecule[j]],
             [mp2_biggest_reps[j] for j in range(len(mp2_biggest_reps)) if
              i == molecule[j]], 1, c = colormap(i, n), label = names[i])
plt.title("SCF energy vs. biggest (absolute) MP2 rep, colored by molecule")
plt.xlabel("SCF energy")
plt.ylabel("MP2 rep value")
plt.legend(ncol = 3)
plt.savefig("images/classification/scf_reps_colored.png")

#scf v mp2 energy
#without groups
plt.figure(dpi=200)
plt.scatter(scf_Es, mp2_Es, 1)
plt.title("SCF energy vs. MP2 energy")
plt.xlabel("SCF energy")
plt.ylabel("MP2 energy")
plt.savefig("images/classification/scf_mp2.png")

#with groups
plt.figure(dpi=200)
for i in range(n) :
    plt.scatter([scf_Es[j] for j in range(len(scf_Es)) if i == molecule[j]],
             [mp2_Es[j] for j in range(len(mp2_Es)) if
              i == molecule[j]], 1, c = colormap(i, n), label = names[i])
plt.title("SCF energy vs. MP2 energy, colored by molecule")
plt.xlabel("SCF energy")
plt.ylabel("MP2 energy")
plt.legend(ncol = 3)
plt.savefig("images/classification/scf_mp2_colored.png")
    
#mp2 energy v mp2 reps
#without groups
plt.figure(dpi=200)
plt.scatter(mp2_Es, mp2_biggest_reps, 1)
plt.title("MP2 energy vs. biggest (absolute) MP2 rep")
plt.xlabel("MP2 energy")
plt.ylabel("MP2 rep value")
plt.savefig("images/classification/mp2_reps.png")

#with groups
plt.figure(dpi=200)
for i in range(n) :
    plt.scatter([mp2_Es[j] for j in range(len(mp2_Es)) if i == molecule[j]],
             [mp2_biggest_reps[j] for j in range(len(mp2_biggest_reps)) if
              i == molecule[j]], 1, c = colormap(i, n), label = names[i])
plt.title("MP2 energy vs. biggest (absolute) MP2 rep, colored by molecule")
plt.xlabel("MP2 energy")
plt.ylabel("MP2 rep value")
plt.legend(ncol = 3)
plt.savefig("images/classification/mp2_reps_colored.png")
