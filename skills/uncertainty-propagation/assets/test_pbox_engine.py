import math
import pytest
from scipy import stats
import pbox_engine as p


def test_normal_location_pbox_exact_cdf_bounds():
    r=p.cdf_bounds({'kind':'normal','mean':[-1,1],'std':[1,1]},0)
    assert r['cdf_lower']==pytest.approx(stats.norm.cdf(-1),abs=1e-9)
    assert r['cdf_upper']==pytest.approx(stats.norm.cdf(1),abs=1e-9)


def test_degenerate_pbox_collapses_to_single_cdf():
    r=p.cdf_bounds({'kind':'normal','mean':[2,2],'std':[3,3]},1.5)
    exact=stats.norm.cdf((1.5-2)/3)
    assert r['cdf_lower']==pytest.approx(exact,abs=1e-10)
    assert r['cdf_upper']==pytest.approx(exact,abs=1e-10)


def test_probability_bounds_ordered():
    r=p.probability_interval({'kind':'normal','mean':[-.5,.5],'std':[.8,1.2]},lower=-1,upper=1)
    assert 0<=r['probability_lower']<=r['probability_upper']<=1


def test_propagated_family_identity_tracks_epistemic_mean_range():
    r=p.propagate({'variables':['x'],'families':[{'kind':'normal','mean':[-1,1],'std':[1,1]}],'expression':'x','grid_points':3,'n':8192,'seed':22,'thresholds':[0]})
    assert r['status']=='PASS'
    assert r['mean_lower']==pytest.approx(-1,abs=.04)
    assert r['mean_upper']==pytest.approx(1,abs=.04)
    assert r['cdf_lower'][0] < .2 and r['cdf_upper'][0] > .8


def test_nonlinear_propagation_reports_grid_claim_boundary():
    r=p.propagate({'variables':['x'],'families':[{'kind':'normal','mean':[0,1],'std':[.5,1]}],'expression':'x**2','grid_points':3,'n':4096})
    assert r['quantile_lower'][1] <= r['quantile_upper'][1]
    assert 'explicitly evaluated parameter grid' in r['claim_boundary']


def test_invalid_uniform_family_blocked():
    with pytest.raises(p.InputError): p.cdf_bounds({'kind':'uniform','lower':[0,2],'upper':[1,3]},1)
