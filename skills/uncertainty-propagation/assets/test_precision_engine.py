import math
import precision_engine as p

def test_normal_case_agrees():
    r=p.high_precision_propagate({'variables':['x'],'values':[.2],'covariance':[[.01]],'expression':'exp(x)'},80); assert r['status']=='PASS'

def test_cancellation_is_detected_not_silently_repaired():
    r=p.high_precision_propagate({'variables':['x'],'values':[1e-30],'covariance':[[1e-62]],'expression':'(1+x)-1','precision_relative_tolerance':1e-12},100)
    assert r['status']=='FLOAT_DISAGREEMENT'; assert abs(float(r['high_precision_nominal'][0])-1e-30)/1e-30 < 1e-12

def test_interval_encloses_monotone_exp():
    r=p.interval_evaluate({'variables':['x'],'expression':'exp(x)','interval_bounds':{'x':[0,1]}}); lo,hi=r['intervals'][0];assert lo<=1<=hi;assert hi>=math.e

def test_interval_claim_is_not_coverage():
    r=p.interval_evaluate({'variables':['x'],'expression':'x','interval_bounds':{'x':[-1,2]}});assert 'not a probability' in r['claim_boundary']
