#!/usr/bin/python3

import mlqm
import psi4
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main(**kwargs) :
    pes = mlqm.PES("pes.json")
    opts = {"basis": pes.basis, "scf_type": "pk", "e_convergence": 1e-8,
            "d_convergence": 1e-8}
    extras = """
wfn.to_file('wfn.npy')
import numpy as np
np.save("geom.npy", mol.geometry().to_array())
np.save("charges.npy", np.array([mol.fZ(i) for i in range(mol.natom())]))
"""

    dlist = pes.generate(opts, directory = "./pes", extra = extras,
                         regen = kwargs["regen"])

    pes.save()
    pes.run(progress = True, restart = kwargs["regen"])
    pes.save()
    results = mlqm.datahelper.grabber(dlist, varnames = ["SCF TOTAL ENERGY",
                                                         "CCSD CORRELATION ENERGY"],
                                      fnames = ["geom.npy", "charges.npy"])
    scf_E = [E for E in list(results['SCF TOTAL ENERGY'].values())]
    ccsd_corr = [E for E in list(results['CCSD CORRELATION ENERGY'].values())]
    ccsd_E = np.add(scf_E, ccsd_corr)
    geoms = [m for m in list(results["geom.npy"].values())]
    charges = [m for m in list(results["charges.npy"].values())]
    reps = [mlqm.repgen.make_coulomb(geoms[m], charges[m]).flatten()
            for m in range(len(geoms))]

    ds = mlqm.Dataset(inpf = "ml.json", reps = reps, vals = ccsd_corr)

    if "regen" in kwargs and kwargs["regen"] :
        ds.data['trainers'] = False
        ds.data['hypers'] = False
        ds.data['a'] = False
        ds.data['s'] = False
        ds.data['l'] = False

    validators = list(reversed([i for i in range(0, ds.setup["N"], 4)]))
    valid_reps = [ds.grand["representations"][val] for val in validators]
    valid_vals = [ds.grand["values"][val] for val in validators]
    ds.grand["representations"] = np.delete(ds.grand["representations"], validators,
                                            axis = 0)
    ds.grand["values"] = np.delete(ds.grand["values"], validators, axis = 0)

    ds.find_trainers("k-means")

    ds, t_AVG = ds.train("KRR")
    ds.save()
    pred_corr = mlqm.krr.predict(ds, valid_reps)
    pred_E = np.add(pred_corr, t_AVG)
    pred_E = np.add(pred_E, [scf_E[i] for i in validators])

    pes_x = np.linspace(float(pes.dis[0]), float(pes.dis[1]), int(pes.pts))
    v_pes = [pes_x[i] for i in validators]
    noval_pes = np.delete(pes_x, validators, axis = 0)
    t_pes = [noval_pes[i] for i in ds.data["trainers"]]

    plt.figure(1, dpi = kwargs["dpi"])
    plt.xlabel("Bond length (Å)")
    plt.ylabel("Energy (Hartree)")
    if "title" in kwargs and kwargs["title"] is not None :
        plt.title(kwargs["title"])
    else :
        plt.title("PES Predicted from the Coulomb Matrix")
        
    plt.plot(pes_x, scf_E, 'y-', label = "SCF PES")
    plt.plot(pes_x, ccsd_E, 'b-', label = "CCSD PES")
    plt.plot(v_pes, pred_E, 'r-^', label = "J PES")
    plt.plot(t_pes, [min(min(scf_E), min(pred_E)) * 1.1 for i in range(len(t_pes))],
             "go", label = "Training Points")
    plt.legend()
    plt.savefig("pes.png")
    
    plt.figure(2, dpi = kwargs["dpi"])
    plt.xlabel("Bond length (Å)")
    plt.ylabel("Difference in Predicted Energy (Hartree)")
    plt.title("Learning Error")
    plt.plot(v_pes, np.subtract([ccsd_E[i] for i in validators], pred_E), "bo",
             label = "Absolute Errors")
    plt.plot(pes_x, [0.002 for i in pes_x], "r-")
    plt.plot(pes_x, [-0.002 for i in pes_x], "r-")
    plt.legend()
    plt.savefig("error.png")
    plt.show()

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--regen", type = bool, dest = "regen", default = False,
                        help = "Whether to regenerate the plots.")
    parser.add_argument("--title", type = str, dest = "title", default = None,
                        help = "What the title for the graphs should be.")
    parser.add_argument("--dpi", type = int, dest = "dpi", default = 200,
                        help = "The desired dpi of the output images.")
    args = parser.parse_args()
    main(**vars(args))
