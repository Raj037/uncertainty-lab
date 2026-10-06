import pytest
from scipy import stats
import time_reliability_engine as t


def test_brownian_bridge_matches_exact_reflection_principle():
    r=t.brownian({'x0':0,'drift':0,'sigma':1,'barrier':1,'horizon':1,'steps':80,'n':80000,'seed':2})
    exact=2*stats.norm.sf(1)
    assert r['exact_failure_probability']==pytest.approx(exact,abs=1e-12)
    assert abs(r['failure_probability']-exact) < 5*r['standard_error']+.004


def test_brownian_drift_formula_and_scale_invariance():
    a=t.brownian({'x0':0,'drift':.2,'sigma':1,'barrier':1.2,'horizon':2,'steps':100,'n':60000,'seed':11})
    b=t.brownian({'x0':0,'drift':.2e-20,'sigma':1e-20,'barrier':1.2e-20,'horizon':2,'steps':100,'n':60000,'seed':11})
    assert b['exact_failure_probability']==pytest.approx(a['exact_failure_probability'],rel=1e-12)
    assert b['failure_probability']==pytest.approx(a['failure_probability'],abs=1e-12)


def test_ou_grid_probability_increases_with_horizon():
    a=t.ou({'long_run_mean':0,'mean_reversion':1,'stationary_std':1,'x0':0,'barrier':2,'horizon':1,'steps':100,'n':30000,'seed':9})
    b=t.ou({'long_run_mean':0,'mean_reversion':1,'stationary_std':1,'x0':0,'barrier':2,'horizon':3,'steps':300,'n':30000,'seed':9})
    assert b['failure_probability'] > a['failure_probability']
    assert 'monitoring grid' in b['claim_boundary']


def test_ar1_stronger_persistence_changes_crossing_probability():
    a=t.ar1({'phi':0.0,'mean':0,'stationary_std':1,'x0':0,'barrier':2,'steps':20,'n':30000,'seed':4})
    b=t.ar1({'phi':0.8,'mean':0,'stationary_std':1,'x0':0,'barrier':2,'steps':20,'n':30000,'seed':4})
    assert 0<a['failure_probability']<1 and 0<b['failure_probability']<1
    assert a['failure_probability'] != pytest.approx(b['failure_probability'],abs=.01)


def test_lower_barrier_supported_by_sign_mapping():
    a=t.brownian({'x0':0,'drift':0,'sigma':1,'barrier':-1,'failure_side':'lower','horizon':1,'steps':80,'n':40000,'seed':6})
    b=t.brownian({'x0':0,'drift':0,'sigma':1,'barrier':1,'failure_side':'upper','horizon':1,'steps':80,'n':40000,'seed':6})
    assert a['failure_probability']==pytest.approx(b['failure_probability'],abs=1e-12)
