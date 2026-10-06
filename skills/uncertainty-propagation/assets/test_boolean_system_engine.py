import math
import pytest
from scipy import stats
import boolean_system_engine as b


def base(n=120000,seed=3):
    return {'variables':['x','y'],'joint_model':{'kind':'multivariate_normal','mean':[0,0],'covariance':[[1,0],[0,1]]},'n':n,'seed':seed,
            'components':{'A':'2-x','B':'2-y'}}


def test_series_and_parallel_match_independent_truth():
    p=base(); pa=stats.norm.sf(2)
    s=b.evaluate({**p,'system_event':'A or B'}); q=b.evaluate({**p,'system_event':'A and B'})
    assert s['system_failure_probability']==pytest.approx(1-(1-pa)**2,abs=.002)
    assert q['system_failure_probability']==pytest.approx(pa**2,abs=.0005)


def test_shared_input_dependence_preserved_not_independence_formula():
    p={'variables':['x'],'joint_model':{'kind':'multivariate_normal','mean':[0],'covariance':[[1]]},'n':100000,'seed':8,
       'components':{'A':'1-x','B':'1.2-x'},'system_event':'A or B'}
    r=b.evaluate(p)
    # union is simply x>=1 because A contains B; independence product would be wrong
    exact=stats.norm.sf(1)
    assert r['system_failure_probability']==pytest.approx(exact,abs=.003)
    pa=r['component_failure_probability']['A'];pb=r['component_failure_probability']['B'];ind=1-(1-pa)*(1-pb)
    assert abs(r['system_failure_probability']-ind)>.01


def test_general_boolean_and_minimal_cut_sets():
    p={'variables':['x'],'joint_model':{'kind':'multivariate_normal','mean':[0],'covariance':[[1]]},'n':5000,'seed':1,
       'components':{'A':'x','B':'x','C':'x'},'system_event':'A or (B and C)'}
    r=b.evaluate(p)
    assert sorted(r['minimal_cut_sets'])==sorted([['A'],['B','C']])


def test_not_operator_supported():
    p=base(n=10000);r=b.evaluate({**p,'system_event':'A and not B'})
    assert 0<=r['system_failure_probability']<=1


def test_unknown_component_fails_closed():
    with pytest.raises(b.InputError): b.evaluate({**base(n=1000),'system_event':'A or Z'})
