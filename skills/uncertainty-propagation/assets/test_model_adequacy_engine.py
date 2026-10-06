import numpy as np
import pytest
import model_adequacy_engine as m


def test_good_model_not_rejected_but_not_proven():
    y=np.array([0.08,-0.12,0.03,0.04,-0.05,0.09,-0.02,0.01])
    p={'observed':y.tolist(),'predicted':[0]*len(y),'standard_uncertainties':[0.2]*len(y),
       'effects':[{'name':'temperature','status':'included','material':True}],'effect_register_complete':True,'effect_register_justification':'measurement procedure and influence-quantity review completed'}
    r=m.assess(p)
    assert r['status']=='MODEL_READY'
    assert r['adequacy_conclusion']=='ADEQUACY_NOT_REJECTED_NOT_PROVEN'
    assert not r['residual_diagnostics']['lack_of_fit_rejected']


def test_strong_bias_is_flagged_provisional():
    p={'observed':[3.0]*10,'predicted':[0.0]*10,'standard_uncertainties':[0.2]*10,'effects':[]}
    r=m.assess(p)
    assert r['status']=='PROVISIONAL'
    assert r['adequacy_conclusion']=='MODEL_INADEQUACY_EVIDENCE'
    assert r['residual_diagnostics']['chi_square_pvalue'] < 1e-20


def test_unresolved_material_effect_prevents_ready():
    p={'observed':[0.01,-0.01,0.0,0.01],'predicted':[0]*4,'standard_uncertainties':[0.1]*4,
       'effects':[{'name':'humidity','status':'unresolved','material':True}]}
    r=m.assess(p)
    assert r['status']=='PROVISIONAL'
    assert r['adequacy_conclusion']=='ADEQUACY_UNRESOLVED'


def test_discrepancy_ml_detects_excess_scatter():
    obs=np.array([-2,-1.5,-1,0,1,1.5,2.0])
    p={'observed':obs.tolist(),'predicted':[0]*len(obs),'standard_uncertainties':[0.1]*len(obs)}
    r=m.estimate_additive_discrepancy(p)
    assert r['tau_hat'] > 0.5
    assert r['log_likelihood_gain'] > 10


def test_compare_models_prefers_closer_prediction():
    y=np.linspace(-1,1,20)
    r=m.compare_models({'observed':y.tolist(),'standard_uncertainties':[0.1]*20,'models':[
        {'name':'good','predicted':y.tolist(),'parameter_count':1},
        {'name':'bad','predicted':(y+0.5).tolist(),'parameter_count':1}]})
    good,bad=r['models']
    assert good['AICc'] < bad['AICc']
    assert good['AICc_weight'] > 0.999


def test_invalid_covariance_fails_closed():
    with pytest.raises(m.InputError):
        m.residual_diagnostics({'observed':[1,2],'predicted':[1,2],'covariance':[[1,2],[2,1]]})


def test_empty_or_unjustified_effect_register_cannot_prove_readiness():
    p={'observed':[0.01,-0.01,0,0.01],'predicted':[0]*4,'standard_uncertainties':[.1]*4,'effects':[]}
    r=m.assess(p)
    assert r['status']=='PROVISIONAL'
    assert r['adequacy_conclusion']=='EFFECT_REGISTER_COMPLETENESS_NOT_ESTABLISHED'
