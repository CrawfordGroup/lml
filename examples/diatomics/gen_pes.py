#!/usr/bin/python3

import psi4
import mlqm
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import concurrent.futures
import argparse

extras = """
wfn.to_file('wfn.npy')
import mlqm
import numpy as np
mp2_amps = mlqm.datahelper.get_amps(wfn, 'MP2')
np.save('mp2_amps.npy', mp2_amps['t2'])
"""

#Wrap it all up into a nice little function.
def main(path, **kwargs) :
    #Go into the directory.
    if "print" in kwargs and kwargs['print'] :
        print("Going into %s/{path}"%(os.getcwd()))
    os.chdir(path)
    try :
        pes = mlqm.PES('input.json')
        if pes.complete and os.path.isfile('images/%s'%(pes.name)) :
            if "print" in kwargs and kwargs['print'] :
                print("Leaving %s"%(os.getcwd()))
            os.chdir('..')
            return
    except FileNotFoundError :
        if "print" in kwargs and kwargs['print'] :
            print("Leaving %s"%(os.getcwd))
        os.chdir('..')
        return
    
    dlist = pes.generate({'basis': pes.basis,
        'scf_type': 'pk',
        'mp2_type': 'conv',
        'e_convergence': 1e-8,
        'd_convergence': 1e-8}, directory = 'pes', extra = extras)
    pes.save()

    pes.run(progress=(False if "print" not in kwargs else kwargs['print']))
    pes.save()

    results = mlqm.datahelper.grabber(dlist,
                                      varnames = ['MP2 CORRELATION ENERGY',
                                                  'HF TOTAL ENERGY',
                                                  'CCSD CORRELATION ENERGY'],
                                      fnames = ['mp2_amps.npy'])
    scf_E = [E for E in list(results['HF TOTAL ENERGY'].values())]
    mp2_corr = [E for E in list(results['MP2 CORRELATION ENERGY'].values())]
    ccsd_corr = [E for E in list(results['CCSD CORRELATION ENERGY'].values())]
    mp2_E = np.add(mp2_corr, scf_E)
    ccsd_E = np.add(ccsd_corr, scf_E)
    mp2_reps = [mlqm.repgen.make_tatr("MP2",
                                     results['mp2_amps.npy'][d]) for d in dlist]
    pes_dom = np.linspace(pes.dis[0], pes.dis[1], pes.pts)
    plt.figure(dpi = 200)
    plt.plot(pes_dom, scf_E, 'y-', label = 'SCF PES')
    plt.plot(pes_dom, mp2_E, 'r--', label = 'MP2 PES')
    plt.plot(pes_dom, ccsd_E, 'b-', label = 'CCSD PES')
    plt.title("Potential energy surface of %s"%(pes.name))
    plt.xlabel("Bond length (Å)")
    plt.ylabel("Energy (Hartree)")
    plt.minorticks_on()
    plt.legend()
    plt.savefig('pes.png')

    if "print" in kwargs and kwargs['print'] :
        print("Leaving %s"%(os.getcwd()))
    os.chdir("..")
    plt.savefig('images/%s_pes.png'%(pes.name))
    plt.close()

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--multi', action = 'store_true')

    args = parse.parse_args()
    
    top_level = next(os.walk(top="."))
    if 'images' not in top_level[1] :
        os.mkdir('images')
    if args.multi :
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
        futures = [executor.submit(main, n) for n in top_level[1]]
        concurrent.futures.wait(futures)
    else :
        for file in top_level[1] :
            main(file, print=True)
        
