import numpy as np
import pytest
import v07_engine as v


def test_parent_identity():
    r=v.assert_parent_identity();assert set(r)=={'v06_engine','v06_release_gate','joint_distribution_engine','autodiff_engine','posterior_diagnostics'}


def test_model_adequacy_routing_and_certificate():
    r=v.run_request({'operation':'model_adequacy','observed':[.01,-.01,.0,.01],'predicted':[0]*4,'standard_uncertainties':[.1]*4,'effects':[],'effect_register_complete':True,'effect_register_justification':'declared complete after model review'})
    assert r['status']=='MODEL_READY';assert r['certificate']['engine_version']=='0.7.0'


def test_form_routing():
    r=v.run_request({'operation':'FORM','variables':['x'],'mean':[0],'covariance':[[1]],'limit_state':'3-x'})
    assert r['beta']==pytest.approx(3,abs=1e-8)


def test_pce_routing():
    x=np.linspace(-1,1,40)[:,None];y=x[:,0]**2
    r=v.run_request({'operation':'pce_surrogate','X':x.tolist(),'y':y.tolist(),'marginals':[{'kind':'uniform','lower':-1,'upper':1}],'total_degree':2,
                     'qualification':{'max_nrmse':1e-9,'min_r2':.9999999}})
    assert r['status']=='SURROGATE_QUALIFIED'


def test_parent_v06_operation_still_routes():
    r=v.run_request({'operation':'high_precision_propagate','variables':['x'],'values':[.2],'covariance':[[.01]],'expression':'exp(x)','dps':60})
    assert r['status']=='PASS';assert r['certificate']['engine_version']=='0.6.0'
