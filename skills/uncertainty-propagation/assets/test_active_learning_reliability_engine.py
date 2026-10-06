import numpy as np
import pytest
from scipy import stats
import active_learning_reliability_engine as a

def base():
    X=np.array([[-3.],[-2.],[-1.],[0.],[1.],[2.],[4.]])
    g=3-X[:,0]
    return {'X':X.tolist(),'g':g.tolist(),'variables':['x'],'input_distribution':{'variables':['x'],'joint_model':{'kind':'multivariate_normal','mean':[0.0],'covariance':[[1.0]]}},'candidate_n':4096,'candidate_seed':7,'gp_noise_std':1e-8}

def test_selector_requests_boundary_point():
    r=a.select_points({**base(),'batch_size':2,'u_stop_threshold':4.0,'max_classification_interval_width':1e-5})
    assert r['status'] in {'NEEDS_EVALUATIONS','RELIABILITY_CONVERGED'}
    pts=np.asarray(r['next_candidate_points'])[:,0]
    assert np.min(np.abs(pts-3)) < .5

def test_expression_active_learning_matches_normal_tail():
    r=a.active_learn_expression({**base(),'oracle_expression':'3-x','evaluation_budget':8,'u_stop_threshold':2.0,'max_classification_interval_width':.003})
    exact=float(stats.norm.sf(3))
    assert r['failure_probability_surrogate_mean_classifier']==pytest.approx(exact,abs=.002)
    assert r['classification_probability_lower'] <= r['failure_probability_surrogate_mean_classifier'] <= r['classification_probability_upper']

def test_fixed_seed_is_deterministic():
    p={**base(),'oracle_expression':'3-x','evaluation_budget':3}
    assert a.active_learn_expression(p)==a.active_learn_expression(p)
