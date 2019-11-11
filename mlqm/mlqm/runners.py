#!/usr/bin/python3

import psi4
import numpy as np
from . import base
from . import output
import concurrent.futures
import os
import subprocess

"""
runners

Contains various adapters to different QC programs.

Contributor: Connor Briggs

Classes
-------
Psi4Runner
    Runs using Psi4.
"""

class Psi4Runner(base.Runner) :
    """
    Psi4Runner

    Runs an input file using Psi4.

    Contributor: Connor Briggs

    Methods
    -------
    __init__(self)
        Default constructor.

    getsingleton()
        See base.Singleton.getsingleton

    run_item(self, directory, inpf, outpf, **kwargs)
        Runs an input file through Psi4.

    run
        Inherited from super.
    """

    def __init__(self) :
        pass

    def run_item(self, inpf, outpf, **kwargs) :
        """
        Psi4Runner.run_item

        Runs a single input file through Psi4.

        Contributor: Connor Briggs

        Parameters
        ----------
        inpf
            The name of the input file

        outpf
            The name of the output file

        mediator
            Reference to the OutputMediator, if given.
        """
        subprocess.call(['psi4', inpf])
        # Command the progress bar to increment.
        if "mediator" in kwargs :
            kwargs["mediator"].submit(output.ProgressBarCommand("inc"))
            kwargs["mediator"].submit(output.ProgressBarCommand("write"))

    def run(self, directories, inpf = "input.dat", outpf = "output.dat",
            executor = None, *args, **kwargs) :
        """
        Psi4Runner.run

        Overridden here to add default values for inpf and outpf,
        as well as for a progress bar.

        Contributor: Connor Briggs

        Paramters
        ---------
        See Runner.run

        See Also
        --------
        base.Runner.run
        """
        super().run(directories, inpf, outpf, executor = executor,
                    *args, **kwargs)
