import numpy as np
import pytest
from scipy import stats
import conformity_engine as c


def test_one_sided_conformance_probability_normal():
    r=c.conformance_probability({'tolerance_upper':10,'measurement_distribution':{'kind':'normal','mean':9.8,'std':.2}})
    assert r['conformance_probability']==pytest.approx(float(stats.norm.cdf(1.0)),abs=1e-12)


def test_guard_band_one_sided_matches_normal_quantile():
    r=c.design_guard_band({'tolerance_upper':10,'standard_uncertainty':.2,'max_specific_consumer_risk':.025})
    assert r['status']=='PASS'
    assert r['guard_band_width']==pytest.approx(float(stats.norm.ppf(.975)*.2),rel=1e-12)
    assert r['consumer_risk_at_active_acceptance_boundary']==pytest.approx(.025,abs=1e-12)


def test_item_decision_reports_specific_consumer_risk():
    p={'tolerance_upper':10,'acceptance_upper':9.6,'measured_value':9.5,'measurement_distribution':{'kind':'normal','mean':9.5,'std':.2}}
    r=c.decide(p)
    assert r['decision']=='ACCEPT'
    assert r['specific_consumer_risk_if_accepted']==pytest.approx(1-float(stats.norm.cdf(2.5)),abs=1e-12)


def test_two_sided_guard_can_be_impossible_for_large_uncertainty():
    r=c.design_guard_band({'tolerance_lower':-1,'tolerance_upper':1,'standard_uncertainty':2,'max_specific_consumer_risk':.01})
    assert r['status']=='NO_NONEMPTY_ACCEPTANCE_INTERVAL_MEETS_TARGET'


def test_global_risk_simulation_matches_direct_logic_and_is_deterministic():
    p={'tolerance_lower':-1,'tolerance_upper':1,'acceptance_lower':-.8,'acceptance_upper':.8,'n':100000,'seed':99,
       'true_distribution':{'kind':'normal','mean':0,'std':1},'measurement_error_distribution':{'kind':'normal','mean':0,'std':.2}}
    a=c.global_risk(p);b=c.global_risk(p)
    assert a==b
    assert 0<a['global_consumer_risk_joint_probability_false_accept']<.1
    assert 0<a['global_producer_risk_joint_probability_false_reject']<.2


def test_expanded_acceptance_fails_closed_without_optin():
    with pytest.raises(c.InputError):
        c.decide({'tolerance_lower':0,'tolerance_upper':1,'acceptance_lower':-.1,'acceptance_upper':1.1,'measured_value':.5,
                  'measurement_distribution':{'kind':'normal','mean':.5,'std':.1}})
