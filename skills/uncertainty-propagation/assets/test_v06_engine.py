import numpy as np
import pytest
import v06_engine as v6


def test_parent_identity_is_exact():
    got=v6.assert_parent_identity()
    assert got==v6.EXPECTED_PARENT_HASHES


def test_v06_new_operations_are_certified():
    r=v6.run_request({'operation':'precision_verify','variables':['x'],'values':[.2],'covariance':[[.01]],'expression':'exp(x)','dps':80})
    assert r['status']=='PASS'
    assert r['certificate']['engine_version']=='0.6.0'
    assert r['certificate']['parent_identity']==v6.EXPECTED_PARENT_HASHES


def test_v05_operation_is_preserved():
    r=v6.run_request({'operation':'gaussian_coverage_region','mean':[0,0],'covariance':[[1,0],[0,1]],'probability':.95})
    assert r['engine_version']=='0.5.0'
    assert 'certificate' in r


def test_posterior_operation_routes():
    rng=np.random.default_rng(1)
    chains=rng.normal(size=(4,1000,1))
    r=v6.run_request({'operation':'posterior_diagnostics','chains':chains.tolist()})
    assert r['status']=='QUALIFIED'
    assert r['certificate']['engine_version']=='0.6.0'
