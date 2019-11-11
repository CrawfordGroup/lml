#!/usr/bin/python3

"""
Example implementation of the code using strict object-oriented programming.
This is a very strict way of doing it, but it gets the point across. A hybrid
approach using what we already have may be better, but this is at least a start.
"""

import mlqm
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import json
import os
import shutil

if __name__ == "__main__" :
    try :
        os.mkdir("example_pes")
    except :
        pass

    # Read the data file.
    f = open("example.json")
    dic = json.load(f)
    f.close()

    # Get singleton references.
    runner = mlqm.runners.Psi4Runner.getsingleton()
    ingen = mlqm.inputgen.Psi4InputGenerator.getsingleton()
    builder = mlqm.molsets.PESBuilder.getsingleton()
    mediator = mlqm.output.OutputMediator.getsingleton()
    out = mlqm.output.Outputter.getsingleton()

    # Setup the output evironment.
    mediator.register(mlqm.output.OutputCommand, out)

    # Create an executor for parallel processing.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers = 50)

    # Create the set of molecules from the specification.
    mols = builder.build(dic)

    # Submit a line of output to be printed.
    mediator.submit(mlqm.output.OutputCommand("Finished building PES"))

    # Create the input files.
    dirs = ingen.generate(mols, os.getcwd() + "/example_pes", dic["method"], {
        "basis": dic["basis"]}, regen = True, include_meta = True)
    mediator.submit(mlqm.output.OutputCommand("Finished generating output files"))

    # Run the input files to get the values for energy.
    mediator.submit(mlqm.output.OutputCommand("Running input files"))
    runner.run(dirs, executor = executor, regen = True, mediator = mediator)
    mediator.submit(mlqm.output.OutputCommand("Finished running input files"))

    # Get the data from the output files.
    data = mlqm.datahelper.grabber(dirs, varnames = ["HF TOTAL ENERGY",
                                                     "MP2 CORRELATION ENERGY"],
                                   fnames = ["meta.npy"])
    mediator.submit(mlqm.output.OutputCommand("Finished grabbing data"))

    # Close the output mediator.
    mediator.submit(mlqm.output.EndMediatorCommand())

    # Create plots.
    plt.figure(1, dpi = 200)
    pes_x = [data["meta.npy"][d][0] for d in dirs]
    scf_E = [data["HF TOTAL ENERGY"][d] for d in dirs]
    mp2_E = [data["MP2 CORRELATION ENERGY"][d] + data["HF TOTAL ENERGY"][d]
             for d in dirs]

    plt.plot(pes_x, scf_E, "y-", label="HF PES")
    plt.plot(pes_x, mp2_E, "r-", label = "MP2 PES")
    plt.legend()
    plt.savefig("example_plot.png")
