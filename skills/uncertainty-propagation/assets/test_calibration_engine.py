import numpy as np
import pytest
import calibration_engine as c


def analytic_scalar(y,obs_var,prior_var=1,prior_mean=0):
    n=len(y);v=1/(1/prior_var+n/obs_var);m=v*(prior_mean/prior_var+sum(y)/obs_var);return m,v


def test_linear_gaussian_matches_closed_form():
    y=[1.0,1.2,.8,1.1,.9];A=np.ones((5,1));obs_sd=.5
    r=c.linear_gaussian({'observed':y,'design_matrix':A.tolist(),'prior_mean':[0.0],'prior_covariance':[[1.0]],'observation_sd':obs_sd})
    m,v=analytic_scalar(y,obs_sd**2)
    assert r['posterior_mean'][0]==pytest.approx(m,abs=1e-12)
    assert r['posterior_covariance'][0][0]==pytest.approx(v,abs=1e-12)


def test_explicit_discrepancy_increases_posterior_uncertainty():
    y=[1]*6;A=np.ones((6,1));base={'observed':y,'design_matrix':A.tolist(),'prior_mean':[0.0],'prior_covariance':[[1.0]],'observation_sd':.2}
    a=c.linear_gaussian(base);D=(.3**2*np.eye(6)).tolist();b=c.linear_gaussian({**base,'discrepancy_covariance':D})
    assert b['posterior_sd'][0] > a['posterior_sd'][0]
    assert b['model_discrepancy_covariance'][0][0]==pytest.approx(.09)


def test_metropolis_matches_linear_gaussian_posterior():
    y=[1.0,1.2,.8,1.1,.9,1.05,.95,1.1]
    exact=c.linear_gaussian({'observed':y,'design_matrix':np.ones((len(y),1)).tolist(),'prior_mean':[0.0],'prior_covariance':[[1.0]],'observation_sd':.5})
    r=c.metropolis({'parameter_names':['theta'],'priors':{'theta':{'kind':'normal','mean':0,'std':1}},'observed':y,'observation_sd':.5,
                    'model_expression':'theta','chains':4,'warmup':1200,'draws':2200,'proposal_sd':[.18],'seed':22,
                    'min_bulk_ess':300,'min_tail_ess':300})
    assert r['status']=='POSTERIOR_QUALIFIED'
    assert r['posterior_mean'][0]==pytest.approx(exact['posterior_mean'][0],abs=.04)
    assert r['posterior_sd'][0]==pytest.approx(exact['posterior_sd'][0],abs=.03)
    assert all(.15 < x < .8 for x in r['acceptance_rates'])


def test_vector_data_expression():
    x=np.linspace(-1,1,12);y=2*x+1
    r=c.metropolis({'parameter_names':['a','b'],'priors':{'a':{'kind':'normal','mean':0,'std':5},'b':{'kind':'normal','mean':0,'std':5}},
        'observed':y.tolist(),'observation_sd':.15,'data':{'x':x.tolist()},'model_expression':'a*x+b','chains':4,'warmup':1000,'draws':1800,
        'proposal_sd':[.08,.06],'seed':7,'min_bulk_ess':250,'min_tail_ess':250})
    assert r['status']=='POSTERIOR_QUALIFIED'
    assert r['posterior_mean'][0]==pytest.approx(2,abs=.08)
    assert r['posterior_mean'][1]==pytest.approx(1,abs=.06)


def test_invalid_discrepancy_fails_closed():
    with pytest.raises(c.InputError):
        c.linear_gaussian({'observed':[1,2],'design_matrix':[[1],[1]],'prior_mean':[0],'prior_covariance':[[1]],'observation_sd':.1,
                           'discrepancy_covariance':[[1,2],[2,1]]})
