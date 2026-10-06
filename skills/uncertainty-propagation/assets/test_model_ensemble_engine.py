import math
import numpy as np
import pytest
import model_ensemble_engine as m


def test_mixture_moments_separate_within_and_between():
    p={'models':[{'name':'A','mean':[0.0],'covariance':[[1.0]],'weight':.5},{'name':'B','mean':[2.0],'covariance':[[1.0]],'weight':.5}]}
    r=m.combine(p)
    assert r['mixture_mean'][0]==pytest.approx(1.0)
    assert r['within_model_covariance'][0][0]==pytest.approx(1.0)
    assert r['between_model_covariance'][0][0]==pytest.approx(1.0)
    assert r['total_covariance'][0][0]==pytest.approx(2.0)


def test_log_evidence_weights_softmax_with_prior():
    r=m.combine({'models':[{'name':'A','mean':[0],'covariance':[[1]],'log_evidence':0},{'name':'B','mean':[1],'covariance':[[1]],'log_evidence':math.log(3)}], 'prior_model_weights':[.5,.5]})
    assert r['weights']==pytest.approx([.25,.75],abs=1e-12)
    assert r['mixture_mean'][0]==pytest.approx(.75)


def test_event_probability_mixture():
    r=m.combine({'models':[{'mean':[0],'covariance':[[1]],'weight':.2,'event_probability':.1},{'mean':[0],'covariance':[[1]],'weight':.8,'event_probability':.5}]})
    assert r['mixture_event_probability']==pytest.approx(.42)


def test_robust_weight_interval_bounds_exact():
    p={'models':[{'name':'A','mean':[0],'covariance':[[1]]},{'name':'B','mean':[10],'covariance':[[1]]}], 'weight_intervals':[[.2,.8],[.2,.8]],'quantity':'output_mean'}
    r=m.robust_scalar_bounds(p)
    assert r['lower']==pytest.approx(2.0)
    assert r['upper']==pytest.approx(8.0)


def test_infeasible_weight_intervals_blocked():
    with pytest.raises(m.InputError):m.robust_scalar_bounds({'models':[{'mean':[0],'covariance':[[1]]},{'mean':[1],'covariance':[[1]]}], 'weight_intervals':[[.8,1],[.8,1]]})
