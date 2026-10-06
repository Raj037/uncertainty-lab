import math
import pytest
from scipy import stats
import system_reliability_engine as s

def base(kind):
    return {'variables':['x','y'],'joint_model':{'kind':'multivariate_normal','mean':[0,0],'covariance':[[1,0],[0,1]]},'limit_states':['2-x','2-y'],'system':{'kind':kind},'power':14,'replicates':8,'seed':8}

def test_series_independent_normal_matches_exact():
    r=s.randomized_qmc(base('series'));p=float(stats.norm.sf(2));exact=1-(1-p)**2
    assert r['failure_probability']==pytest.approx(exact,abs=.001)

def test_parallel_independent_normal_matches_exact():
    r=s.randomized_qmc(base('parallel'));p=float(stats.norm.sf(2));exact=p*p
    assert r['failure_probability']==pytest.approx(exact,abs=.0004)

def test_k_out_of_n_two_of_two_equals_parallel():
    a=s.randomized_qmc(base('parallel'));p=base('k_out_of_n');p['system']['k']=2;b=s.randomized_qmc(p)
    assert a['failure_probability']==b['failure_probability']
