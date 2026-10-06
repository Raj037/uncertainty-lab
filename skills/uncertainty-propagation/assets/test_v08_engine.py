import pytest
import v08_engine as v

def test_parent_identity():
    assert set(v.assert_parent_identity())=={'v07_engine','v07_release_gate','reliability_engine','surrogate_engine','calibration_engine'}

def test_experiment_design_routing_and_certificate():
    r=v.run_request({'operation':'experiment_design_rank','parameter_mean':[0],'parameter_covariance':[[1]],'candidates':[{'name':'m','sensitivity':[1],'noise_std':.2}],'objective':'information_gain'})
    assert r['best_candidate']['name']=='m';assert r['certificate']['engine_version']=='0.8.0'

def test_parent_v07_route_remains_available():
    r=v.run_request({'operation':'FORM','variables':['x'],'mean':[0],'covariance':[[1]],'limit_state':'3-x'})
    assert r['beta']==pytest.approx(3,abs=1e-8);assert r['certificate']['engine_version']=='0.7.0'
