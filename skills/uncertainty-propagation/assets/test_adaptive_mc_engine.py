import adaptive_mc_engine as m

def base(**kw):
    p={'variables':['x'],'joint_model':{'kind':'multivariate_normal','mean':[0.0],'covariance':[[1.0]]},'expression':'x','seed':123,'batch_size':10000,'min_samples':40000,'max_samples':200000,'mc_abs_tolerance':0.03,'mc_rel_tolerance':0.02}
    p.update(kw);return p

def test_converges_and_reports_numerical_error():
    r=m.propagate(base()); assert r['status']=='CONVERGED'; assert r['n_generated']>=40000
    assert abs(r['mean'][0])<.03 and abs(r['std'][0]-1)<.03
    assert all(r['convergence'][k]['passed'][0] for k in ['mean','std','qlo','qhi'])

def test_strict_tolerance_fails_closed_at_ceiling():
    r=m.propagate(base(max_samples=40000,mc_abs_tolerance=1e-8,mc_rel_tolerance=1e-8)); assert r['status']=='NOT_CONVERGED';assert r['reason']=='MAX_SAMPLES_REACHED'

def test_deterministic_same_seed():
    a=m.propagate(base());b=m.propagate(base());assert a==b

def test_invalid_configuration_blocks():
    try:m.propagate(base(batch_size=100,min_samples=200))
    except m.InputError:return
    assert False
