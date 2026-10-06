import pytest
import robust_decision_engine as r


def test_degenerate_weights_reduce_to_bayes_decision():
    p={'actions':['A','B'],'scenarios':['s1','s2'],'loss_matrix':[[1,5],[3,2]],'weight_intervals':[[.4,.4],[.6,.6]],'nominal_weights':[.4,.6]}
    z=r.analyze(p)
    assert z['bayes_action']==z['minimax_expected_loss_action']


def test_minimax_can_differ_from_nominal_bayes():
    p={'actions':['risky','safe'],'scenarios':['good','bad'],'loss_matrix':[[0,10],[4,4]],'weight_intervals':[[.2,.9],[.1,.8]],'nominal_weights':[.9,.1]}
    z=r.analyze(p)
    assert z['bayes_action']=='risky'
    assert z['minimax_expected_loss_action']=='safe'


def test_minimax_regret_reported():
    p={'actions':['A','B','C'],'scenarios':['s1','s2'],'loss_matrix':[[0,8],[4,4],[8,0]],'weight_intervals':[[.2,.8],[.2,.8]]}
    z=r.analyze(p)
    assert z['minimax_regret_action']=='B'


def test_robust_design_chooses_best_worst_case_utility():
    z=r.robust_utility_design({'candidates':['specialist','balanced'],'scenarios':['m1','m2'],'utility_matrix':[[10,0],[5,5]],'weight_intervals':[[.2,.8],[.2,.8]]})
    assert z['best_candidate']['candidate']=='balanced'
