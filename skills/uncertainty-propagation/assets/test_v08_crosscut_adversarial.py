import numpy as np
import pytest
import hmc_nuts_engine as h
import system_reliability_engine as s
import discrepancy_inference_engine as d
import experimental_design_engine as e

def test_nuts_parameter_unit_scaling_regression():
    y=[1.0,1.2,.8,1.1,.9];base={'parameter_names':['theta'],'priors':{'theta':{'kind':'normal','mean':0,'std':1}},'observed':y,'observation_sd':.5,'model_expression':'theta','chains':4,'warmup':250,'draws':350,'seed':4,'min_bulk_ess':50,'min_tail_ess':50,'max_treedepth':6}
    a=h.nuts(base);sc=1e-20;b=h.nuts({**base,'priors':{'theta':{'kind':'normal','mean':0,'std':sc}},'observed':(np.array(y)*sc).tolist(),'observation_sd':.5*sc})
    assert b['posterior_mean'][0]/sc==pytest.approx(a['posterior_mean'][0],abs=.05)

def test_system_reliability_preserves_common_cause_dependence():
    p={'variables':['x','y'],'joint_model':{'kind':'multivariate_normal','mean':[0,0],'covariance':[[1,.8],[.8,1]]},'limit_states':['1.5-x','1.5-y'],'system':{'kind':'series'},'power':13,'replicates':6,'seed':7}
    r=s.randomized_qmc(p);ind=r['individual_event_probabilities'];independent_union=1-(1-ind[0])*(1-ind[1])
    assert abs(r['failure_probability']-independent_union)>.005

def test_discrepancy_identifiability_invariant_to_design_column_scale():
    n=12;K=np.ones((n,n));p={'observed':np.ones(n).tolist(),'design_matrix':np.ones((n,1)).tolist(),'prior_mean':[0.0],'prior_covariance':[[1]],'observation_sd':.1,'discrepancy_kernel':K.tolist(),'tau_prior_scale':.4}
    a=d.infer_linear_gaussian(p);b=d.infer_linear_gaussian({**p,'design_matrix':(np.ones((n,1))*1e-20).tolist(),'prior_covariance':[[1e40]]})
    assert a['parameter_discrepancy_confounding_fraction']==pytest.approx(b['parameter_discrepancy_confounding_fraction'],abs=1e-12)

def test_design_ranking_invariant_under_parameter_unit_change():
    p={'parameter_mean':[0.0],'parameter_covariance':[[4.0]],'candidates':[{'name':'m','sensitivity':[1.0],'noise_std':.2}],'objective':'information_gain'}
    a=e.score_candidates(p);sc=1e-20;b=e.score_candidates({'parameter_mean':[0.0],'parameter_covariance':[[4*sc*sc]],'candidates':[{'name':'m','sensitivity':[1/sc],'noise_std':.2}],'objective':'information_gain'})
    assert a['best_candidate']['information_gain_nats']==pytest.approx(b['best_candidate']['information_gain_nats'],rel=1e-10)
