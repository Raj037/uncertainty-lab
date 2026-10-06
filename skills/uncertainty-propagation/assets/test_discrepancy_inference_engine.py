import numpy as np
import pytest
import discrepancy_inference_engine as d

def test_iid_discrepancy_recovers_scale_and_parameter():
    rng=np.random.default_rng(2);n=40;theta=1.2;tau=.3;obs=.1;y=theta+rng.normal(scale=(obs**2+tau**2)**.5,size=n)
    r=d.infer_linear_gaussian({'observed':y.tolist(),'design_matrix':np.ones((n,1)).tolist(),'prior_mean':[0.0],'prior_covariance':[[4.0]],'observation_sd':obs,'discrepancy_kernel':np.eye(n).tolist(),'tau_prior_scale':.5})
    assert r['status']=='POSTERIOR_QUALIFIED'
    assert r['posterior_parameter_mean'][0]==pytest.approx(theta,abs=.12)
    assert r['tau_posterior_mean']==pytest.approx(tau,abs=.12)
    assert r['parameter_discrepancy_confounding_fraction'] < .1

def test_aligned_discrepancy_is_flagged_as_confounding():
    n=20;K=np.ones((n,n));y=np.ones(n)
    r=d.infer_linear_gaussian({'observed':y.tolist(),'design_matrix':np.ones((n,1)).tolist(),'prior_mean':[0.0],'prior_covariance':[[1.0]],'observation_sd':.1,'discrepancy_kernel':K.tolist(),'tau_prior_scale':.5,'max_confounding_fraction':.8})
    assert r['status'].startswith('POSTERIOR_PROVISIONAL')
    assert r['parameter_discrepancy_confounding_fraction']==pytest.approx(1.0,abs=1e-12)

def test_invalid_kernel_fails_closed():
    with pytest.raises(d.InputError):d.infer_linear_gaussian({'observed':[1,1,1],'design_matrix':[[1],[1],[1]],'prior_mean':[0],'prior_covariance':[[1]],'observation_sd':.1,'discrepancy_kernel':[[1,2,0],[2,1,0],[0,0,1]]})
