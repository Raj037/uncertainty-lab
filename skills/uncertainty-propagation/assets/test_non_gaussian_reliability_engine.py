import math
import pytest
from scipy import stats
import non_gaussian_reliability_engine as r

def test_qmc_lognormal_exact_tail():
    q=float(stats.lognorm.ppf(.95,s=.4,scale=math.exp(.2)))
    p={'variables':['x'],'joint_model':{'kind':'gaussian_copula','copula_correlation':[[1.0]],'marginals':[{'kind':'lognormal','log_mean':.2,'log_std':.4}]},'limit_state':f'{q}-x','power':13,'replicates':8,'seed':7}
    got=r.randomized_qmc(p)
    assert got['failure_probability']==pytest.approx(.05,abs=.002)
    assert got['standard_error_across_scrambles']<.0015

def test_qmc_multivariate_t_reproducible():
    p={'variables':['x','y'],'joint_model':{'kind':'multivariate_t','mean':[0,0],'scale_matrix':[[1,.3],[.3,1]],'df':5},'limit_state':'3-x-y','power':12,'replicates':4,'seed':11}
    assert r.randomized_qmc(p)==r.randomized_qmc(p)

def test_directional_normal_linear_matches_exact():
    p={'variables':['x'],'joint_model':{'kind':'multivariate_normal','mean':[0.0],'covariance':[[1.0]]},'limit_state':'1.6448536269514722-x','directions':800,'replicates':4,'seed':3}
    got=r.directional(p)
    assert got['failure_probability']==pytest.approx(.05,abs=.004)

def test_directional_lognormal_tail_matches_exact():
    q=float(stats.lognorm.ppf(.95,s=.5,scale=1.0))
    p={'variables':['x'],'joint_model':{'kind':'gaussian_copula','copula_correlation':[[1.0]],'marginals':[{'kind':'lognormal','log_mean':0,'log_std':.5}]},'limit_state':f'{q}-x','directions':800,'replicates':4,'seed':13}
    got=r.directional(p)
    assert got['failure_probability']==pytest.approx(.05,abs=.004)

def test_directional_rejects_t_copula():
    p={'variables':['x'],'joint_model':{'kind':'t_copula','df':5,'copula_correlation':[[1.0]],'marginals':[{'kind':'normal','mean':0,'std':1}]},'limit_state':'2-x'}
    with pytest.raises(r.InputError): r.directional(p)

def test_qmc_refuses_implicit_joint_model():
    with pytest.raises(r.InputError):r.randomized_qmc({'variables':['x'],'limit_state':'2-x'})
