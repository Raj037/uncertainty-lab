import v10_engine as v

def test_parent_identity():
 r=v.assert_parent_identity();assert 'v09_engine' in r and 'pbox_engine' in r

def test_model_ensemble_route_certificate():
 r=v.run_request({'operation':'model_ensemble','models':[{'mean':[0],'covariance':[[1]],'weight':.5},{'mean':[2],'covariance':[[1]],'weight':.5}]})
 assert r['mixture_mean'][0]==1;assert r['certificate']['engine_version']=='1.0.0'

def test_parent_v09_route_still_routes():
 r=v.run_request({'operation':'pbox_cdf','pbox':{'kind':'normal','mean':[-1,1],'std':[1,1]},'x':0})
 assert r['cdf_lower']<r['cdf_upper'];assert r['certificate']['engine_version']=='0.9.0'
