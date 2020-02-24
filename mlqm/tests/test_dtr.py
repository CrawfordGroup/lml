import numpy as np
import mlqm

def test_dtr():
    D = np.load('./outputs/h2o_D.npy')
    dtr = mlqm.repgen.make_dtr(D)
    ref = np.load('./outputs/h2o_dtr.npy')

    assert np.allclose(dtr,ref), "DTR does not match reference."

if __name__ == "__main__":
    test_dtr()
