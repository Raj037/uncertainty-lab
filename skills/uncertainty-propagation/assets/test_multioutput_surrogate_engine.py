import numpy as np
import pytest
import multioutput_surrogate_engine as m

def test_multioutput_pce_exact_covariance():
    rng=np.random.default_rng(1);X=rng.normal(size=(240,2));Y=np.column_stack([X[:,0]+X[:,1],2*X[:,0]])
    r=m.pce({'X':X.tolist(),'Y':Y.tolist(),'marginals':[{'kind':'normal','mean':0,'std':1}]*2,'total_degree':1,'validation_fraction':.2,'validation_repeats':3,'qualification':{'max_nrmse':1e-10,'min_r2':.999999999}})
    assert r['status']=='SURROGATE_QUALIFIED'
    assert np.array(r['output_covariance'])==pytest.approx(np.array([[2.,2.],[2.,4.]]),abs=1e-9)

def test_multioutput_gp_retains_shared_input_covariance():
    x=np.linspace(-2,2,100)[:,None];Y=np.column_stack([x[:,0],2*x[:,0]])
    r=m.gp_propagate({'X':x.tolist(),'Y':Y.tolist(),'validation_fraction':.2,'validation_repeats':2,'noise_std':1e-6,'qualification':{'max_nrmse':.02,'min_r2':.99},'input_distribution':{'variables':['x'],'joint_model':{'kind':'multivariate_normal','mean':[0.0],'covariance':[[1.0]]}},'n':10000,'seed':4})
    assert r['status']=='SURROGATE_QUALIFIED'
    C=np.array(r['input_driven_output_covariance'])
    assert C[0,1]>1.8 and C[1,1]>3.6
