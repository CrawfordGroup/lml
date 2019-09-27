#!/usr/bin/python3

import json
import os
import shutil
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--regen', action = 'store_true')
args = parse.parse_args()

#Cation definitions for ionic compounds.
cat1 = ['Li', 'Na', 'K']
cat2 = ['Be', 'Mg', 'Ca']
#Anion definitions
ani1 = ['H', 'F', 'Cl', 'Br']
ani2 = ['O']

#Covalent compounds.
cov1 = ['H', 'F', 'Cl', 'Br']
cov_others = [['C', 'O'], ['O', 'O'], ['N', 'N']]

n = 200
longest = 2.0

#Make inputs for diatomic ionic compounds.
for m in cat1 :
    for x in ani1 :
        try :
            os.mkdir("%s%s"%(m, x))
        except :
            if args.regen :
                shutil.rmtree('%s%s'%(m, x))
                os.mkdir('%s%s'%(m, x))
            else :
                pass
        if args.regen or not os.path.isfile('%s%s/input.json'%(m, x)):
            fd = open('%s%s/input.json'%(m, x), 'w+')
            json.dump({'inpf': 'input.json', 'name' : '%s%s'%(m, x),
                        'geom': '%s\n%s 1 x'%(m, x), 'method': 'CCSD',
                        'pts': n, 'gvar': 'x', 'dis': [0.5, longest],
                        'basis': 'sto-3g', 'generated': False,
                        'complete': False}, fd)
            fd.close()
            fd = open('%s%s/ml.json'%(m, x), 'w+')
            json.dump({'setup': {'name': '%s%s'%(m, x), 'valtype': 'CCSD',
                                 'predtype': 'MP2', 'M': 12, 'N': n,
                                 'st': 0.5, 'K': 30}, 'data': {
                                     'trainers': False, 'hypers': False,
                                     's': False, 'l': False, 'a': False}},
                      fd)
            fd.close()

for m in cat2 :
    for x in ani2 :
        try :
            os.mkdir('%s%s'%(m, x))
        except :
            if args.regen :
                shutil.rmtree('%s%s'%(m, x))
                os.mkdir('%s%s'%(m, x))
            else :
                pass
        if args.regen or not os.path.isfile('%s%s/input.json'%(m, x)):
            fd = open('%s%s/input.json'%(m, x), 'w+')
            json.dump({'inpf': '%s%s/input.json'%(m, x), 'name' : '%s%s'%(m, x),
                        'geom': '%s\n%s 1 x'%(m, x), 'method': 'CCSD',
                        'pts': n, 'gvar': 'x', 'dis': [0.5, longest],
                        'basis': 'sto-3g', 'generated': False,
                        'complete': False}, fd)
            fd.close()
            fd = open('%s%s/ml.json'%(m, x), 'w+')
            json.dump({'setup': {'name': '%s%s'%(m, x), 'valtype': 'CCSD',
                                 'predtype': 'MP2', 'M': 12, 'N': n,
                                 'st': 0.5, 'K': 30}, 'data': {
                                     'trainers': False, 'hypers': False,
                                     's': False, 'l': False, 'a': False}},
                      fd)
            fd.close()

for i in range(len(cov1)) :
    for j in range(i, len(cov1)) :
        if cov1[i] == cov1[j] :
            name = '%s2'%(cov1[i])
        else :
            name = '%s%s'%(cov1[j], cov1[i])
        try :
            os.mkdir(name)
        except :
            if args.regen :
                shutil.rmtree(name)
                os.mkdir(name)
            else :
                pass
        if args.regen or not os.path.isfile('%s/input.json'%(name)):
            fd = open('%s/input.json'%(name), 'w+')
            json.dump({'inpf': 'input.json',
                        'name' : name,
                        'geom': '%s\n%s 1 x'%(cov1[j], cov1[i]),
                        'method': 'CCSD',
                        'pts': n, 'gvar': 'x', 'dis': [0.5, longest],
                        'basis': 'sto-3g', 'generated': False,
                        'complete': False}, fd)
            fd.close()
            fd = open('%s/ml.json'%(name), 'w+')
            json.dump({'setup': {'name': name, 'valtype': 'CCSD',
                                 'predtype': 'MP2', 'M': 12, 'N': n,
                                 'st': 0.5, 'K': 30}, 'data': {
                                     'trainers': False, 'hypers': False,
                                     's': False, 'l': False, 'a': False}},
                      fd)
            fd.close()
        
for c in cov_others :
    if c[0] == c[1] :
        name = '%s2'%(c[0])
    else :
        name = '%s%s'%(c[0], c[1])
    try :
        os.mkdir(name)
    except :
        if args.regen :
            shutil.rmtree(name)
            os.mkdir(name)
        else :
            pass
    if args.regen or not os.path.isfile('%s/input.json'%(name)):
        fd = open('%s/input.json'%(name), 'w+')
        json.dump({'inpf': 'input.json', 'name' : name,
                    'geom': '%s\n%s 1 x'%(c[0], c[1]), 'method': 'CCSD',
                    'pts': n, 'gvar': 'x', 'dis': [0.5, longest],
                    'basis': 'def2-svp', 'generated': False,
                    'complete': False}, fd)
        fd.close()
        fd = open('%s/ml.json'%(name), 'w+')
        json.dump({'setup': {'name': name, 'valtype': 'CCSD',
                             'predtype': 'MP2', 'M': 12, 'N': n,
                             'st': 0.5, 'K': 30}, 'data': {
                                 'trainers': False, 'hypers': False,
                                 's': False, 'l': False, 'a': False}},
                  fd)
        fd.close()
