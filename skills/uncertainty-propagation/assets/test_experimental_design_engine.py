import numpy as np
import pytest
import experimental_design_engine as e

def base():
    return {'parameter_mean':[0,0],'parameter_covariance':[[4,0],[0,1]],'candidates':[{'name':'measure_a','sensitivity':[1,0],'noise_std':.1,'cost':1},{'name':'measure_b','sensitivity':[0,1],'noise_std':.1,'cost':1},{'name':'mixed','sensitivity':[1,1],'noise_std':.5,'cost':3}]}

def test_information_gain_prefers_high_variance_direction():
    r=e.score_candidates({**base(),'objective':'information_gain'})
    assert r['best_candidate']['name']=='measure_a'

def test_target_variance_prefers_target_direction():
    r=e.score_candidates({**base(),'objective':'target_variance_reduction','target_sensitivity':[0,1]})
    assert r['best_candidate']['name']=='measure_b'

def test_decision_entropy_prefers_target_measurement():
    r=e.score_candidates({**base(),'objective':'decision_entropy_reduction','target_sensitivity':[0,1],'decision_threshold':0.0})
    assert r['best_candidate']['name']=='measure_b'
    assert r['best_candidate']['decision_entropy_reduction_nats']>0

def test_cost_normalization_can_change_choice():
    p=base();p['candidates'][0]['cost']=100
    r=e.score_candidates({**p,'objective':'information_gain','per_cost':True})
    assert r['best_candidate']['name']!='measure_a'

def test_greedy_batch_selects_distinct_and_reduces_covariance():
    r=e.greedy_batch({**base(),'objective':'information_gain','batch_size':2})
    assert len(set(r['selected_candidates']))==2
    assert np.trace(np.array(r['posterior_covariance_after_batch'])) < 5
