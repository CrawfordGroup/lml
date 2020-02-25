import numpy as np
from mlqm import repgen as rg

def test_coulomb():
    ref = np.array([[73.51669472,  4.23341769,  4.23341769],
                    [ 4.23341769,  0.5       ,  0.33463019],
                    [ 4.23341769,  0.33463019,  0.5       ]])

    geom = np.array([[ 0.        ,  0.        , -0.12947689],
                     [ 0.        , -1.49418674,  1.0274461 ],
                     [ 0.        ,  1.49418674,  1.0274461 ]])

    charges = np.array([8.0,1.0,1.0])

    ref_sym = ref[np.tril_indices(len(ref))]

    assert np.allclose(rg.make_coulomb(geom,charges),ref), "Coulomb matrix does not match reference."

    assert np.allclose(rg.make_coulomb(geom,charges,ignore_matrix_symmetry=False), ref_sym), "Symmetry-reduced Coulomb elements do not match reference."
