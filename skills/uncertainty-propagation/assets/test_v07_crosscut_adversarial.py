import numpy as np
import pytest
import model_adequacy_engine as adequacy
import reliability_engine as reliability
import surrogate_engine as surrogate
import calibration_engine as calibration
import conformity_engine as conformity


def test_adequacy_chi_square_is_invariant_to_output_unit_rescaling():
    obs=np.array([1.1,.9,1.05,.95]);pred=np.ones(4);u=np.full(4,.1)
    base=adequacy.residual_diagnostics({'observed':obs.tolist(),'predicted':pred.tolist(),'standard_uncertainties':u.tolist()})
    for scale in [1e-30,1e30]:
        r=adequacy.residual_diagnostics({'observed':(obs*scale).tolist(),'predicted':(pred*scale).tolist(),'standard_uncertainties':(u*scale).tolist()})
        assert r['chi_square']==pytest.approx(base['chi_square'],rel=1e-12)
        assert r['chi_square_pvalue']==pytest.approx(base['chi_square_pvalue'],rel=1e-12)


def test_form_reliability_is_invariant_to_physical_unit_scaling():
    a=reliability.form({'variables':['x'],'mean':[0],'covariance':[[1]],'limit_state':'4-x'})
    b=reliability.form({'variables':['x'],'mean':[0],'covariance':[[1e-40]],'limit_state':'4e-20-x'})
    assert b['beta']==pytest.approx(a['beta'],rel=1e-10)
    assert b['failure_probability_FORM']==pytest.approx(a['failure_probability_FORM'],rel=1e-10)


def test_underfit_pce_is_rejected_by_holdout_gate():
    x=np.linspace(-1,1,100)[:,None];y=x[:,0]**2
    r=surrogate.polynomial_chaos({'X':x.tolist(),'y':y.tolist(),'marginals':[{'kind':'uniform','lower':-1,'upper':1}],
        'total_degree':1,'validation_repeats':3,'qualification':{'max_nrmse':.01,'min_r2':.99}})
    assert r['status']=='SURROGATE_REJECTED'


def test_calibration_never_invents_discrepancy_covariance():
    p={'observed':[1,1.1,.9],'design_matrix':[[1],[1],[1]],'prior_mean':[0],'prior_covariance':[[1]],'observation_sd':.2}
    a=calibration.linear_gaussian(p)
    assert np.max(np.abs(np.array(a['model_discrepancy_covariance'])))==0
    d=.3**2*np.eye(3);b=calibration.linear_gaussian({**p,'discrepancy_covariance':d.tolist()})
    assert np.allclose(np.array(b['model_discrepancy_covariance']),d)


def test_guarded_acceptance_shrinks_not_expands_tolerance():
    r=conformity.design_guard_band({'tolerance_lower':0,'tolerance_upper':10,'standard_uncertainty':.5,'max_specific_consumer_risk':.05})
    assert r['status']=='PASS'
    assert 0 < r['acceptance_lower'] < r['acceptance_upper'] < 10
