import numpy as np
import mlqm

def test_tatr():
    amps = mlqm.datahelper.harvest_amps('MP2',namps=500,outfile="./tests/outputs/h2o_out.dat")
    tatr = mlqm.repgen.make_tatr('mp2',amps['t2'])
    ref = np.load('./tests/outputs/h2o_tatr.npy')

    assert np.allclose(tatr,ref), "TATR does not match reference."

if __name__ == "__main__":
    test_tatr()
