#!/usr/bin/python3

import mlqm.molecule
def test_all() :
    # Create a simple molecule.
    geom = [("H", 1, 0, 0, 0), ("H", 1, 0, 0, 0.74)]
    mol = mlqm.molecule.Molecule("Test", [("H", 1, 0, 0, 0),
                                          ("H", 1, 0, 0, 0.74)], 0, 1)
    assert all(all(mol.geometry()[i][j] == geom[i][j] for j in
               range(len(geom[i]))) for i in range(len(geom)))

    assert mol.charge() == 0
    assert mol.spinstate() == 1
    assert str(mol) == "Test"
