import numpy as np
import mlqm

def test_amp_harvest():
    amps = mlqm.datahelper.harvest_amps('mp2',namps=150,outfile='./outputs/less_amps.out')
    ref = np.array([-0.0473461109, 
                    -0.0330907442,
                    -0.0305718680,
                    -0.0299288042,
                     0.0295057294,
                     0.0295057294,
                    -0.0243683773,
                    -0.0243683773,
                    -0.0240952855,
                    -0.0230489846])

    assert np.allclose(amps['t2'],ref), "Amplitudes do not match reference."

if __name__ == "__main__":
    test_amp_harvest()
