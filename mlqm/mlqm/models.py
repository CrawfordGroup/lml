#!/usr/bin/python3

from sklearn import kernel_ridge, gaussian_process
from . import core
from . import datahelper
import numpy as np

def model(name) :
    if name.lower() in ["krr", "kernel-ridge", "kernel_ridge", "kernel ridge"] :
        return KRRModel
    elif name.lower() in ["gpr", "gaussian-process", "gaussian_process",
                          "gaussian process"] :
        return GPRModel

class GPRModel(core.RegressionModel) :
    def train(ds, **kwargs) :
            
