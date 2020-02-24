import numpy as np
import mlqm
import sklearn as skl
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

def test_krr_wrap():
    reps = np.load('datasets/h2_coul/coulombs.npy')
    mp2_E = np.load('datasets/h2_coul/mp2_E.npy')

    # train on first 10 points, predict remaining 90
    trainers = [i for i in range(0,10)]
    t_reps = reps[trainers]
    t_mp2_E = mp2_E[trainers]
    v_reps = np.delete(reps,trainers,axis=0)
    v_mp2_E = np.delete(mp2_E,trainers,axis=0)

    # MLQM (default):
    ## rbf kernel
    ## k-fold CV (k = M)
    ## alpha/gamma hypers opt on np.logspace(-12,12,num=50)
    ## neg_mean_squared_error loss
    ## training targets automatically shifted by average
    setup = {
            "name": "h2_coul",
            "M": 10,
            "N": 100,
            }
    data = {
           "trainers": trainers,
           "hypers": False,
           "s": False,
           "l": False,
           "a": False
           }
    ds = mlqm.Dataset(reps = reps, vals = mp2_E)
    ds.setup = setup
    ds.data = data
    ds,t_AVG = ds.train("KRR")
    a_pred = mlqm.krr.predict(ds,v_reps)
    a_pred = np.add(a_pred,t_AVG)

    # SKL 
    ## use MLQM defaults, manually shift training targets
    krr = KernelRidge(kernel='rbf')
    avg = np.mean(t_mp2_E)
    parameters = {'alpha':np.logspace(-12,12,num=50),
                  'gamma':np.logspace(-12,12,num=50)}
    krr_regressor = GridSearchCV(krr,parameters,scoring='neg_mean_squared_error',cv=len(t_mp2_E))
    krr_regressor.fit(t_reps,t_mp2_E-avg)
    krr = krr_regressor.best_estimator_
    b_pred = krr.predict(v_reps) + avg

    assert np.allclose(a_pred,b_pred), "MLQM predictions do not match SKL predictions."

if __name__ == "__main__" :
    test_krr_wrap()


