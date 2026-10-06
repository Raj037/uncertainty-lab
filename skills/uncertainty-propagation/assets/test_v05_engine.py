import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import v05_engine as v

def test_research_linear_model_ready_and_exact_covariance():
    p={'variables':['x','y'],'values':[1,2],'covariance':[[.04,.01],[.01,.09]],'expression':'x+2*y'};r=v.research_propagate(p)
    assert r['status']=='MODEL_READY';assert r['verification']['status']=='PASS';assert abs(r['analytic']['first_order_output_covariance'][0][0]-(.04+4*.09+4*.01))<1e-12

def test_nonlinearity_is_provisional_not_silently_linearized():
    r=v.research_propagate({'variables':['x'],'values':[0.],'covariance':[[.25]],'expression':'exp(x)'})
    assert r['status']=='PROVISIONAL';assert r['analytic']['nonlinearity_triggers']

def test_tiny_exp_numerical_disagreement_blocks():
    r=v.research_propagate({'variables':['x'],'values':[1e-20],'covariance':[[1e-40]],'expression':'exp(x)'})
    assert r['status']=='BLOCKED';assert r['issue']=='THREE_ENGINE_DISAGREEMENT'

def test_vector_gaussian_coverage_is_retained():
    r=v.research_propagate({'variables':['x','y'],'values':[0,0],'covariance':[[1,.3],[.3,2]],'expressions':['x+y','x-y'],'coverage_probability':.95})
    assert r['gaussian_coverage_region']['rank']==2

def test_dispatch_joint_non_gaussian():
    r=v.run_request({'operation':'joint_propagate','variables':['x','y'],'joint_model':{'kind':'gaussian_copula','copula_correlation':[[1,.5],[.5,1]],'marginals':[{'kind':'uniform','lower':0,'upper':1},{'kind':'gamma','shape':2,'scale':1}]},'n':10000,'seed':1,'expression':'x*y'})
    assert r['status']=='ok'

def test_certificate_deterministic_and_input_sensitive():
    p={'operation':'research_propagate','variables':['x'],'values':[2.0],'covariance':[[.04]],'expression':'x*x'}
    a=v.run_request(p);b=v.run_request(p);assert a['certificate']==b['certificate']
    q=dict(p);q['values']=[2.1];c=v.run_request(q);assert a['certificate']['payload_sha256']!=c['certificate']['payload_sha256']

def test_certificate_binds_all_v05_source_modules():
    p={'operation':'gaussian_coverage_region','mean':[0,0],'covariance':[[1,0],[0,1]],'probability':.95}
    r=v.run_request(p);src=r['certificate']['source_sha256'];assert 'v05_engine' in src and 'coverage_regions' in src

def test_release_gate_operation_passes_all_new_capabilities():
    r=v.run_request({'operation':'v05_release_gate','seed':20261005})
    assert r['status']=='PASS'
    assert len(r['rows'])==8
    assert all(x['status']=='PASS' for x in r['rows'])
    assert r['certificate']['engine_version']=='0.5.0'
