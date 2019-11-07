#!/usr/bin/python3

import mlqm
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import json
import os
import shutil

if __name__ == "__main__" :
    shutil.rmtree("example_pes")
    f = open("example.json")
    dic = json.load(f)
    f.close()
    
    runner = mlqm.runners.Psi4Runner.getsingleton()
    ingen = mlqm.inputgen.Psi4InputGenerator.getsingleton()
    builder = mlqm.molsets.PESBuilder.getsingleton()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers = 50)

    mols = builder.build(dic)
    print("Built molecule set")
    dirs = ingen.generate(mols, "./example_pes", dic["method"], {
        "basis": dic["basis"]}, regen = True, include_meta = True)
    print("Generated input files")
    runner.run(dirs, executor = None, regen = True)
    print("Finished running files")
    data = mlqm.datahelper.grabber(dirs, varnames = ["HF TOTAL ENERGY",
                                                     "MP2 CORRELATION ENERGY"],
                                   fnames = ["meta.npy"])

    plt.figure(1, dpi = 200)
    pes_x = [data["meta.npy"][d][0] for d in dirs]
    scf_E = [data["HF TOTAL ENERGY"][d] for d in dirs]
    mp2_E = [data["MP2 CORRELATION ENERGY"][d] + data["HF TOTAL ENERGY"][d]
             for d in dirs]

    plt.plot(pes_x, scf_E, "y-", label="HF PES")
    plt.plot(pes_x, mp2_E, "r-", label = "MP2 PES")
    plt.legend()
    plt.savefig("example_plot.png")
