import mlqm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

opts = {'basis':'6-311g',
        'scf_type':'pk',
        'mp2_type':'conv',
        'freeze_core':'false',
        'e_convergence':1e-8,
        'd_convergence':1e-8}

# Let's write the wfn and the MP2 amplitudes to file for later!
extra_stuff = """
wfn.to_file('wfn.npy')
import mlqm
import numpy as np
mp2_amps = mlqm.datahelper.get_amps(wfn,'MP2')
np.save('mp2_amps.npy',mp2_amps['t2'])
"""

# Build PES object with settings
pes = mlqm.PES('pes.json')

# Generate input files and hold directory structure
dlist = pes.generate(opts,directory='./h2_pes',extra=extra_stuff)
pes.save()

# Run all input files on the PES
pes.run()
pes.save()

# Pull results from output json and custom files in directory structure
results = mlqm.datahelper.grabber(dlist,
            varnames=['MP2 CORRELATION ENERGY',
                      'HF TOTAL ENERGY'],
            fnames=['mp2_amps.npy'])
scf_E = [E for E in list(results['HF TOTAL ENERGY'].values())]
mp2_E = np.add([E for E in list(results['MP2 CORRELATION ENERGY'].values())],scf_E)

# Let's try making MP2 TATRs out of the saved amplitudes
mp2_reps = [mlqm.repgen.make_tatr('MP2',results['mp2_amps.npy'][d]) for d in dlist]

# Graph one of the TATRs!
plt.figure(2,dpi=200)
plt.plot(mp2_reps[25])
plt.savefig('tatr.png')

# Graph the PESs!
pes = np.linspace(0.5,2.0,50)
plt.figure(1,dpi=200)
plt.plot(pes,scf_E,'y-',label='SCF PES')
plt.plot(pes,mp2_E,'r--',label='MP2 PES')
plt.legend()
plt.savefig('pes.png')
