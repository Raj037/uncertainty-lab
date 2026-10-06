import numpy as np
import pytest
import global_design_engine as g
import experimental_design_engine as e


def test_global_linear_batch_is_at_least_as_good_as_greedy_information_gain():
    p={'parameter_mean':[0,0,0],'parameter_covariance':[[4,1,0],[1,2,.3],[0,.3,1]],'objective':'information_gain','batch_size':2,
       'candidates':[{'name':'a','sensitivity':[1,0,0],'noise_std':.5},{'name':'b','sensitivity':[0,1,0],'noise_std':.5},{'name':'c','sensitivity':[0,0,1],'noise_std':.5},{'name':'mix','sensitivity':[1,1,0],'noise_std':.8}]}
    glob=g.exact_global_batch(p);gre=e.greedy_batch(p)
    assert glob['best_batch']['information_gain_nats']+1e-12 >= gre['total_information_gain_nats']
    assert glob['evaluated_combinations']==6


def test_nonlinear_eig_prefers_informative_candidate():
    rng=np.random.default_rng(1);theta=rng.normal(size=(1200,1))
    r=g.nonlinear_eig({'parameter_names':['theta'],'parameter_samples':theta.tolist(),'outer_samples':300,'replicates':4,'seed':4,
        'candidates':[{'name':'informative','model_expression':'theta','noise_std':.3},{'name':'weak','model_expression':'0.05*theta','noise_std':1.0}]})
    assert r['best_candidate']['name']=='informative'
    assert r['ranked_candidates'][0]['expected_information_gain_nats'] > r['ranked_candidates'][1]['expected_information_gain_nats']


def test_cost_normalization_can_change_ranking():
    rng=np.random.default_rng(2);theta=rng.normal(size=(1000,1))
    base={'parameter_names':['theta'],'parameter_samples':theta.tolist(),'outer_samples':250,'replicates':3,'seed':7,
          'candidates':[{'name':'precise_expensive','model_expression':'theta','noise_std':.15,'cost':10},{'name':'cheap','model_expression':'theta','noise_std':.5,'cost':1}]}
    raw=g.nonlinear_eig(base);pc=g.nonlinear_eig({**base,'per_cost':True})
    assert raw['best_candidate']['name']=='precise_expensive'
    assert pc['best_candidate']['name']=='cheap'


def test_global_nonlinear_batch_deterministic_and_returns_best_pair():
    rng=np.random.default_rng(3);X=rng.normal(size=(800,2))
    p={'parameter_names':['a','b'],'parameter_samples':X.tolist(),'batch_size':2,'outer_samples':180,'replicates':3,'seed':11,
       'candidates':[{'name':'a','model_expression':'a','noise_std':.4},{'name':'b','model_expression':'b','noise_std':.4},{'name':'redundant_a','model_expression':'a','noise_std':.8}]}
    r1=g.nonlinear_global_batch(p);r2=g.nonlinear_global_batch(p)
    assert r1==r2
    assert set(r1['best_batch']['candidates'])=={'a','b'}


def test_exact_global_target_objective():
    p={'parameter_mean':[0,0],'parameter_covariance':[[4,0],[0,1]],'objective':'target_variance_reduction','target_sensitivity':[0,1],'batch_size':1,
       'candidates':[{'name':'measure_a','sensitivity':[1,0],'noise_std':.2},{'name':'measure_b','sensitivity':[0,1],'noise_std':.2}]}
    r=g.exact_global_batch(p)
    assert r['best_batch']['candidates']==['measure_b']
