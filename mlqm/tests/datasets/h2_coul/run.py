#!/usr/bin/python3

import mlqm
import psi4
import matplotlib.pyplot as plt
import numpy as np

pes = mlqm.PES("pes.json")
opts = {"basis": pes.basis, "scf_type": "pk", "e_convergence": 1e-8,
        "d_convergence": 1e-8}
extras = """
import numpy as np
np.save("geom.npy", mol.geometry().to_array())
np.save("charges.npy", np.array([mol.fZ(i) for i in range(mol.natom())]))
"""

dlist = pes.generate(opts, directory = "./pes", extra = extras)

pes.save()
pes.run(progress = True)
pes.save()
results = mlqm.datahelper.grabber(dlist, varnames = ["SCF TOTAL ENERGY",
                                                     "MP2 CORRELATION ENERGY"],
                                  fnames = ["geom.npy", "charges.npy"])
scf_E = [E for E in list(results['SCF TOTAL ENERGY'].values())]
mp2_corr = [E for E in list(results['MP2 CORRELATION ENERGY'].values())]
geoms = [m for m in list(results["geom.npy"].values())]
charges = [m for m in list(results["charges.npy"].values())]
reps = [mlqm.repgen.make_coulomb(geoms[m], charges[m]).flatten()
        for m in range(len(geoms))]

np.save('mp2_E.npy',mp2_corr)
np.save('scf_E.npy',scf_E)
np.save('coulombs.npy',reps)

