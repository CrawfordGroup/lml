#!/usr/bin/python3
import mlqm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import os
import util
import math
import argparse

#find the average square residue between
#the predicted value and the actual value.
def standard_error(actual, other) :
    return sum((other[i] - actual[i]) ** 2 for i in
               range(len(other))) / len(other)

if __name__ == "__main__" :
    parser = arpgarse.ArgumentParser()
    parser.add_argument("--plots", type=bool,
                        help=
                        "Whether to show the interactive plots when finished with processing.")
    args = parser.parse_args()
    fp = open("sets.json")
    inpf = json.load(fp)
    fp.close()

    pes = {}
    dlist = {}
    results = {}
    scf_E = {}
    mp2_corr = {}
    ccsd_corr = {}
    mp2_reps = {}
    t_avgs = {}
    validators = {}
    valid_reps = {}
    valid_vals = {}
    pebbles = {}
    mp2_reps_all = []
    ccsd_corr_all = []

    extra_stuff = """
    wfn.to_file('wfn.npy')
    import mlqm
    import numpy as np
    mp2_amps = mlqm.datahelper.get_amps(wfn, 'MP2')
    np.save('mp2_amps.npy', mp2_amps['t2'])
    """


    for f in inpf['training'] :
        print(f"Running {f}", flush=True)
        pes[f] = mlqm.PES(f)

        if pes[f].complete and \
           all(os.path.isfile(d) for d in [f'pes/{pes[f].name}/{n}.npy'
                                           for n in ['scf_E', 'mp2_corr',
                                                     'ccsd_corr', 'mp2_reps']]):
            scf_E[f] = np.load(f'pes/{pes[f].name}/scf_E.npy')
            mp2_corr[f] = np.load(f'pes/{pes[f].name}/mp2_corr.npy')
            ccsd_corr[f] = np.load(f'pes/{pes[f].name}/ccsd_corr.npy')
            mp2_reps[f] = np.load(f'pes/{pes[f].name}/mp2_reps.npy')
        else :   
            dlist[f] = pes[f].generate({'basis': pes[f].basis, 'scf_type': 'pk',
                                     'mp2_type': 'conv', 'freeze_core': 'false',
                                     'e_convergence':1e-8, 'd_convergence':1e-8},
                                    directory=pes[f].name, extra = extra_stuff)
            pes[f].save()
            pes[f].run(progress=True)
            pes[f].save()
            results[f] = mlqm.datahelper.grabber(dlist[f], varnames=[
                'MP2 CORRELATION ENERGY', 'HF TOTAL ENERGY', 'CCSD CORRELATION ENERGY'],
                                                 fnames = ['mp2_amps.npy'])
            scf_E[f] = list(results[f]['HF TOTAL ENERGY'].values())
            mp2_corr[f] = list(results[f]['MP2 CORRELATION ENERGY'].values())
            ccsd_corr[f] = list(results[f]['CCSD CORRELATION ENERGY'].values())

            mp2_reps[f] = [mlqm.repgen.make_tatr('MP2', results[f]['mp2_amps.npy'][d])
                           for d in dlist[f]]

            np.save(f'pes/{pes[f].name}/scf_E.npy', scf_E[f])
            np.save(f'pes/{pes[f].name}/mp2_corr.npy', mp2_corr[f])
            np.save(f'pes/{pes[f].name}/ccsd_corr.npy', ccsd_corr[f])
            np.save(f'pes/{pes[f].name}/mp2_reps.npy', mp2_reps[f])
        mp2_reps_all.extend(mp2_reps[f])
        ccsd_corr_all.extend(ccsd_corr[f])

    for f in inpf['validation'] :
        print(f"Running {f}")
        pes[f] = mlqm.PES(f)

        if pes[f].complete and \
           all(os.path.isfile(d) for d in [f'pes/{pes[f].name}/{n}.npy'
                                           for n in ['scf_E', 'mp2_corr',
                                                     'ccsd_corr', 'mp2_reps']]):
            scf_E[f] = np.load(f'pes/{pes[f].name}/scf_E.npy')
            mp2_corr[f] = np.load(f'pes/{pes[f].name}/mp2_corr.npy')
            ccsd_corr[f] = np.load(f'pes/{pes[f].name}/ccsd_corr.npy')
            mp2_reps[f] = np.load(f'pes/{pes[f].name}/mp2_reps.npy')
        else :
            dlist[f] = pes[f].generate({'basis': pes[f].basis, 'scf_type': 'pk',
                                     'mp2_type': 'conv', 'freeze_core': 'false',
                                     'e_convergence':1e-8, 'd_convergence':1e-8},
                                    directory=pes[f].name, extra = extra_stuff)
            pes[f].save()
            pes[f].run(progress=True)
            pes[f].save()
            results[f] = mlqm.datahelper.grabber(dlist[f], varnames=[
                'MP2 CORRELATION ENERGY', 'HF TOTAL ENERGY', 'CCSD CORRELATION ENERGY'],
                                                 fnames = ['mp2_amps.npy'])
            scf_E[f] = list(results[f]['HF TOTAL ENERGY'].values())
            mp2_corr[f] = list(results[f]['MP2 CORRELATION ENERGY'].values())
            ccsd_corr[f] = list(results[f]['CCSD CORRELATION ENERGY'].values())

            mp2_reps[f] = [mlqm.repgen.make_tatr('MP2', results[f]['mp2_amps.npy'][d])
                           for d in dlist[f]]
            np.save(f'pes/{pes[f].name}/scf_E.npy', scf_E[f])
            np.save(f'pes/{pes[f].name}/mp2_corr.npy', mp2_corr[f])
            np.save(f'pes/{pes[f].name}/ccsd_corr.npy', ccsd_corr[f])
            np.save(f'pes{pes[f].name}/mp2_reps.npy', mp2_reps[f])

    #Train the monolith.
    ds = mlqm.Dataset(inpf = 'ml.json', reps=mp2_reps_all, vals = ccsd_corr_all)
    validators_all = list(reversed(range(0, len(mp2_reps_all), 4)))
    valid_reps_all = [ds.grand['representations'][val] for val in validators_all]
    valid_vals_all = [ds.grand['values'][val] for val in validators_all]
    ds.grand['representations'] = np.delete(ds.grand['representations'],
                                            validators_all,
                                            axis = 0)
    ds.grand['values'] = np.delete(ds.grand['values'], validators_all, axis = 0)
    ds.find_trainers('k-means')
    ds.save()
    print(f"Training set: {ds.data['trainers']}")
    ds, t_AVG = ds.train('KRR')
    print("Trained")
    ds.save()

    #Train the pebbles.
    for f in inpf['training'] :
        print(f"Reading ml_individ_{pes[f].name}.json")
        pebbles[f] = mlqm.Dataset(f"ml_individ_{pes[f].name}.json",
                                  reps = mp2_reps[f], vals = ccsd_corr[f])
        validators[f] = list(reversed(range(0, len(mp2_reps[f]), 4)))
        valid_reps[f] = [pebbles[f].grand['representations'][val] for val
                         in validators[f]]
        valid_vals[f] = [pebbles[f].grand['values'][val] for val in validators[f]]
        pebbles[f].grand['representations'] = np.delete(
            pebbles[f].grand['representations'], validators[f], axis=0)
        pebbles[f].grand['values'] = np.delete(pebbles[f].grand['values'],
                                               validators[f], axis = 0)
        #Just use evenly spaced training points.
        pebbles[f].data["trainers"] = list(map(int, np.linspace(0,
                                                  len(pebbles[f].grand['values']),
                                                  pebbles[f].setup["M"], False)))
        pebbles[f].save()
        print(f"Training set for pebble {pebbles[f].setup['name']}: {pebbles[f].data['trainers']}")
        pebbles[f], t_avgs[f] = pebbles[f].train('KRR')
        print(f"Trained {pebbles[f].setup['name']}")
        pebbles[f].save()

    weight = {}
    #Find the mean square residual.
    for f in inpf["training"] :
        total = 0
        num = 0
        for g in inpf["training"] :
            if g == f :
                continue
            print(f"Predicting {g} from {f}")
            pred = mlqm.krr.predict(pebbles[f], mp2_reps[g])
            total += math.sqrt(util.difference_metrics(pred, ccsd_corr[g], 2) /
                               (len(pred) - 1))
            num += 1
        weight[f] = total / num

    #Define the final predictor.
    stone = lambda reps : sum((np.add(mlqm.krr.predict(pebbles[f], reps),
                                      t_avgs[f])
                               / weight[f]
                               for f in pebbles),
                              np.zeros(len(reps))) / sum(1 / w for w in
                                                         list(weight.values()))

    n = 0
    for f in pes :
        pred_mp2 = mlqm.krr.predict(ds, mp2_reps[f])
        pred2_mp2 = stone(mp2_reps[f])
        pred_E = np.add(pred_mp2, scf_E[f])
        pred_E = np.add(pred_E, t_AVG)
        pred2_E = np.add(pred2_mp2, scf_E[f])
        mp2_E = np.add(mp2_corr[f], scf_E[f])
        ccsd_E = np.add(ccsd_corr[f], scf_E[f])

        pes_axis = np.linspace(pes[f].dis[0], pes[f].dis[1], pes[f].pts)
        #Make error bars for the combined data.
        hold = {}
        for gf in pebbles :
            hold[gf] = np.subtract(
                np.add(mlqm.krr.predict(pebbles[gf], mp2_reps[f]), t_avgs[gf]),
                pred2_mp2)
        diffs = [sorted(hold[gf][i] for gf in hold)
                 for i in range(len(list(hold.values())[0]))]
        error = [[abs(diffs[i][-1]) for i in range(len(diffs))],
                 [abs(diffs[i][0]) for i in range(len(diffs))]]
    
        plt.figure(n, dpi=200)
        plt.plot(pes_axis, scf_E[f], 'y-', label = 'SCF PES')
        plt.plot(pes_axis, mp2_E, 'r--', label = 'MP2 PES')
        plt.plot(pes_axis, ccsd_E, 'b-', label = 'CCSD PES')
        plt.plot(pes_axis, pred_E, 'g.', label = 'Monolith')
        plt.errorbar(pes_axis, pred2_E,
                     yerr = error, fmt='m.', linewidth=1, label = 'Combined',
                     errorevery = 10)
    
        plt.legend()
        n += 1
        if f in inpf['training'] :
            print(f"Writing training/{pes[f].name}.png")
            plt.savefig("training/" + pes[f].name + '.png')
        else :
            print(f"Writing validation/{pes[f].name}.png")
            plt.savefig("validation/" + pes[f].name + '.png')

        plt.figure(n, dpi=200)
        plt.plot(pes_axis, [0.002 for i in pes_axis], 'r-')
        plt.plot(pes_axis, [-0.002 for i in pes_axis], 'r-')
        plt.plot(pes_axis, [(pred_E[i] - ccsd_E[i])
                            for i in range(len(pred_E))], 'b.',
                 label="MP2 Monolith")
        plt.errorbar(pes_axis, [(pred2_E[i] - ccsd_E[i]) for i in
                            range(len(pred2_E))], yerr = error, fmt='g.',
                     label = "MP2 Combined", errorevery = 10,
                     linewidth = 1)
        plt.title(f"Monolith r²: %0.4f, Combined r²: %0.4f"%(
            util.r_squared(pred_E, ccsd_E), util.r_squared(pred2_E, ccsd_E)))
        plt.legend()
        if f in inpf['training'] :
            print(f"Writing training/{pes[f].name}_error.png")
            plt.savefig("training/" + pes[f].name + "_error.png")
        else :
            print(f"Writing validation/{pes[f].name}_error.png")
            plt.savefig("validation/" + pes[f].name + "_error.png")
        n+=1
    if args.plots :
        for f in pes :
            pred_mp2 = mlqm.krr.predict(ds, mp2_reps[f])
            pred2_mp2 = stone(mp2_reps[f])
            pred_E = np.add(pred_mp2, scf_E[f])
            pred_E = np.add(pred_E, t_AVG)
            pred2_E = np.add(pred2_mp2, scf_E[f])
            mp2_E = np.add(mp2_corr[f], scf_E[f])
            ccsd_E = np.add(ccsd_corr[f], scf_E[f])

            pes_axis = np.linspace(pes[f].dis[0], pes[f].dis[1], pes[f].pts)
            #Make error bars for the combined data.
            hold = {}
            for gf in pebbles :
                hold[gf] = np.subtract(
                    np.add(mlqm.krr.predict(pebbles[gf], mp2_reps[f]), t_avgs[gf]),
                    pred2_mp2)
            diffs = [sorted(hold[gf][i] for gf in hold)
                     for i in range(len(list(hold.values())[0]))]
            error = [[abs(diffs[i][-1]) for i in range(len(diffs))],
                     [abs(diffs[i][0]) for i in range(len(diffs))]]
    
            plt.figure(n, dpi=200)
            plt.plot(pes_axis, scf_E[f], 'y-', label = 'SCF PES')
            plt.plot(pes_axis, mp2_E, 'r--', label = 'MP2 PES')
            plt.plot(pes_axis, ccsd_E, 'b-', label = 'CCSD PES')
            plt.plot(pes_axis, pred_E, 'g.', label = 'Monolith')
            plt.errorbar(pes_axis, pred2_E,
                         yerr = error, fmt='m.', linewidth=1, label = 'Combined',
                         errorevery = 10)
    
            plt.legend()
            n += 1
        
            plt.figure(n, dpi=200)
            plt.plot(pes_axis, [0.002 for i in pes_axis], 'r-')
            plt.plot(pes_axis, [-0.002 for i in pes_axis], 'r-')
            plt.plot(pes_axis, [(pred_E[i] - ccsd_E[i])
                                for i in range(len(pred_E))], 'b.',
                     label="MP2 Monolith")
            plt.errorbar(pes_axis, [(pred2_E[i] - ccsd_E[i]) for i in
                                range(len(pred2_E))], yerr = error, fmt='g.',
                         label = "MP2 Combined", errorevery = 10,
                         linewidth = 1)
            plt.title(f"Monolith r²: %0.4f, Combined r²: %0.4f"%(
                util.r_squared(pred_E, ccsd_E), util.r_squared(pred2_E, ccsd_E)))
            plt.legend()
            n+=1
        plt.show()
