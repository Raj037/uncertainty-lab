import pytest
import v09_engine as v

def test_parent_identity():
    r=v.assert_parent_identity();assert 'v08_engine' in r and 'joint_distribution_engine' in r

def test_pbox_route_certificate():
    r=v.run_request({'operation':'pbox_cdf','pbox':{'kind':'normal','mean':[-1,1],'std':[1,1]},'x':0})
    assert r['cdf_lower']<r['cdf_upper'];assert r['certificate']['engine_version']=='0.9.0'

def test_parent_v08_route_still_works():
    r=v.run_request({'operation':'experiment_design_rank','parameter_mean':[0,0],'parameter_covariance':[[2,0],[0,1]],'objective':'information_gain',
                     'candidates':[{'name':'a','sensitivity':[1,0],'noise_std':.2},{'name':'b','sensitivity':[0,1],'noise_std':.2}]})
    assert r['best_candidate']['name']=='a';assert r['certificate']['engine_version']=='0.8.0'
