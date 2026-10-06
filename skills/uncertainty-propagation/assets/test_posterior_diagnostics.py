import numpy as np
import pytest
import posterior_diagnostics as pd


def test_independent_well_mixed_chains_qualify():
    rng=np.random.default_rng(101)
    chains=rng.normal(size=(4,2500,2))
    r=pd.diagnose_mcmc({'chains':chains.tolist(),'parameter_names':['a','b']})
    assert r['status']=='QUALIFIED'
    assert max(x['rhat'] for x in r['parameters']) < 1.01
    assert min(x['bulk_ess'] for x in r['parameters']) > 1000
    assert min(x['tail_ess'] for x in r['parameters']) > 1000


def test_shifted_chain_fails_rhat():
    rng=np.random.default_rng(202)
    chains=rng.normal(size=(4,1800,1))
    chains[3,:,0]+=1.5
    r=pd.diagnose_mcmc({'chains':chains.tolist()})
    assert r['status']=='NOT_QUALIFIED'
    assert r['parameters'][0]['rhat'] > 1.01


def test_highly_autocorrelated_chains_fail_ess():
    rng=np.random.default_rng(303)
    m,n=4,2500
    x=np.zeros((m,n,1))
    for c in range(m):
        eps=rng.normal(size=n)
        for t in range(1,n): x[c,t,0]=.97*x[c,t-1,0]+eps[t]
    r=pd.diagnose_mcmc({'chains':x.tolist(),'min_bulk_ess':400,'min_tail_ess':400})
    assert r['status']=='NOT_QUALIFIED'
    assert r['parameters'][0]['bulk_ess'] < 400


def test_warmup_is_explicitly_discarded():
    rng=np.random.default_rng(404)
    x=rng.normal(size=(4,1200,1))
    x[:,:200,0]+=np.linspace(5,0,200)[None,:]
    r=pd.diagnose_mcmc({'chains':x.tolist(),'warmup':200})
    assert r['warmup_discarded']==200
    assert r['post_warmup_draws_per_chain']==1000


def test_weighted_samples_report_weight_quality_not_mcmc_convergence():
    rng=np.random.default_rng(505)
    s=rng.normal(size=(2000,2));w=np.ones(2000)
    r=pd.diagnose_weighted({'samples':s.tolist(),'weights':w.tolist(),'parameter_names':['x','y']})
    assert r['status']=='WEIGHT_QUALITY_PASS'
    assert r['mcmc_convergence']=='NOT_ASSESSED'
    assert r['weight_ess']==pytest.approx(2000)


def test_degenerate_weights_fail_quality():
    s=np.arange(1000,dtype=float)[:,None];w=np.zeros(1000);w[0]=.9;w[1:]=.1/999
    r=pd.diagnose_weighted({'samples':s.tolist(),'weights':w.tolist(),'min_weight_ess':50,'max_normalized_weight':.1})
    assert r['status']=='WEIGHT_QUALITY_FAIL'
    assert r['max_normalized_weight'] > .8
