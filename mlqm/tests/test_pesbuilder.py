#!/usr/bin/python3

import mlqm

def test() :
    builder = mlqm.molsets.PESBuilder.getsingleton()

    mols1 = builder.build({'geom': "H\nH 1 x", "dis": [0.5, 1.0], "pts": 2,
                           "gvar": 'x'})
    mols2 = builder.build({'geom': "H\nH 1 x", "dis" : [0, 1.5], "pts": 2,
                           "gvar": "x"}, pestype = "exclude")
    # Eventually make this automated.
    for mol in mols1 :
        print(mol.geometry())
    for mol in mols2 :
        print(mol.geometry())

test()
