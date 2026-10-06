import numpy as np
import pytest
import calibration_engine as cal
import hmc_nuts_engine as h

def exact_case():
    y=[1.0,1.2,.8,1.1,.9,1.05,.95,1.1]
    e=cal.linear_gaussian({'observed':y,'design_matrix':np.ones((len(y),1)).tolist(),'prior_mean':[0.0],'prior_covariance':[[1.0]],'observation_sd':.5})
    p={'parameter_names':['theta'],'priors':{'theta':{'kind':'normal','mean':0,'std':1}},'observed':y,'observation_sd':.5,'model_expression':'theta','chains':4,'warmup':500,'draws':900,'seed':12,'min_bulk_ess':200,'min_tail_ess':200}
    return e,p

def test_hmc_matches_exact():
    e,p=exact_case();r=h.hmc({**p,'leapfrog_steps':8})
    assert r['status']=='POSTERIOR_QUALIFIED'
    assert r['posterior_mean'][0]==pytest.approx(e['posterior_mean'][0],abs=.04)
    assert r['posterior_sd'][0]==pytest.approx(e['posterior_sd'][0],abs=.035)
    assert sum(r['post_warmup_divergences'])==0

def test_nuts_matches_exact():
    e,p=exact_case();r=h.nuts({**p,'max_treedepth':7})
    assert r['status']=='POSTERIOR_QUALIFIED'
    assert r['posterior_mean'][0]==pytest.approx(e['posterior_mean'][0],abs=.04)
    assert r['posterior_sd'][0]==pytest.approx(e['posterior_sd'][0],abs=.035)
    assert sum(r['post_warmup_divergent_iterations'])==0

def test_nuts_deterministic_fixed_seed():
    _,p=exact_case();q={**p,'warmup':250,'draws':350,'min_bulk_ess':50,'min_tail_ess':50,'max_treedepth':6}
    assert h.nuts(q)==h.nuts(q)

def test_non_normal_prior_fails_closed():
    _,p=exact_case();p['priors']={'theta':{'kind':'uniform','lower':-2,'upper':2}}
    with pytest.raises(h.InputError):h.nuts(p)


def test_nuts_is_invariant_to_parameter_unit_scale():
    e,p=exact_case();base=h.nuts({**p,'warmup':300,'draws':500,'min_bulk_ess':80,'min_tail_ess':80,'max_treedepth':6})
    scale=1e-20;ys=(np.array(p['observed'])*scale).tolist()
    tiny={'parameter_names':['theta'],'priors':{'theta':{'kind':'normal','mean':0,'std':scale}},'observed':ys,'observation_sd':.5*scale,'model_expression':'theta','chains':4,'warmup':300,'draws':500,'seed':12,'min_bulk_ess':80,'min_tail_ess':80,'max_treedepth':6}
    got=h.nuts(tiny)
    assert got['status']=='POSTERIOR_QUALIFIED'
    assert got['posterior_mean'][0]/scale==pytest.approx(base['posterior_mean'][0],abs=.04)
    assert got['posterior_sd'][0]/scale==pytest.approx(base['posterior_sd'][0],abs=.04)
