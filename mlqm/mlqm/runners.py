#!/usr/bin/python3

import psi4
import numpy as np
from . import base
import concurrent.futures
import os
import subprocess

class Psi4Runner(base.Runner) :
    __singleton = None

    def __init__(self) :
        pass

    def getsingleton() :
        if Psi4Runner.__singleton is None :
            Psi4Runner.__singleton = Psi4Runner()
        return Psi4Runner.__singleton

    def __run_item(directory, inpf, outpf, **kwargs) :
        if os.path.isfile(directory + "/" + outpf) and ("regen" not in kwargs
                                                        or not kwargs['regen']):
            return
        subprocess.call(['psi4', directory + '/' + inpf,
                         '-o' + directory + "/" +  outpf])
        
    def run(self, directories, inpf = "input.dat", outpf= "output.dat",
            executor = None, **kwargs) :
        """
        This method runs a series of files through Psi4. The optional
        parameter 'executor' represents a concurrent.futures.Executor
        object to use to add concurrency to this. If None, then
        the program will execute sequentially.

        Returns
        -------
        void
        """
        if executor is None :
            wd = os.getcwd()
            for d in directories :
                if os.path.isfile(outpf) and ("regen" not in kwargs or
                                              not kwargs['regen']):
                    continue
                subprocess.call(['psi4', d + "/" + inpf, "-o" +
                                 d + "/" + outpf])
        else :
            futures = [executor.submit(Psi4Runner.__run_item, d, inpf, outpf,
                                                    **kwargs) for d in
                                    directories]
            concurrent.futures.wait(futures)
