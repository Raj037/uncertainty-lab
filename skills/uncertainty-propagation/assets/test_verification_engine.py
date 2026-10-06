import sys
from pathlib import Path
import numpy as np,pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import verification_engine as v

def test_linear_randomized_three_engines():
    rng=np.random.default_rng(14)
    for _ in range(100):
        n=5;A=rng.normal(size=(3,n));x=rng.normal(size=n);L=rng.normal(size=(n,n));C=L@L.T+np.eye(n)*.1;names=[f'x{i}' for i in range(n)];expr=['+'.join(f'({A[k,i]:.17g})*x{i}' for i in range(n)) for k in range(3)]
        r=v.triple_verify({'variables':names,'values':x,'covariance':C,'expressions':expr});assert r['status']=='PASS',r['diagnostics']

def test_smooth_nonlinear_three_engines():
    r=v.triple_verify({'variables':['x','y'],'values':[.2,1.3],'covariance':[[.01,.002],[.002,.04]],'expressions':['exp(x)*sin(y)','log(y)+x*x']});assert r['status']=='PASS',r['diagnostics']

def test_tiny_exp_roundoff_is_caught_not_false_passed():
    r=v.triple_verify({'variables':['x'],'values':[1e-20],'covariance':[[1e-40]],'expression':'exp(x)'})
    assert r['status']=='FAIL';assert r['diagnostics']['ad_vs_fd_jacobian']>.9

def test_tiny_linear_control_passes():
    r=v.triple_verify({'variables':['x'],'values':[1e-20],'covariance':[[1e-40]],'expression':'3*x'});assert r['status']=='PASS',r['diagnostics']

def test_hessian_symbolic_ad_match_random_quadratic():
    rng=np.random.default_rng(15);n=4;A=rng.normal(size=(n,n));A=(A+A.T)/2;x=rng.normal(size=n);C=np.eye(n);names=[f'x{i}' for i in range(n)];expr='+'.join(f'({A[i,j]:.17g})*x{i}*x{j}' for i in range(n) for j in range(n))
    r=v.triple_verify({'variables':names,'values':x,'covariance':C,'expression':expr});assert r['status']=='PASS';assert r['diagnostics']['symbolic_vs_ad_hessian']<1e-12

def test_tiny_indefinite_covariance_rejected_scale_invariantly():
    with pytest.raises(v.InputError):v.triple_verify({'variables':['x','y'],'values':[0,0],'covariance':[[1e-40,2e-40],[2e-40,1e-40]],'expression':'x+y'})

def test_extreme_diagonal_unit_rescaling_preserves_verdict():
    C=np.array([[1,.2],[.2,1.5]]);x=np.array([.4,-.3]);D=np.diag([1e-25,1e25]);Cs=D@C@D;xs=D@x
    r=v.triple_verify({'variables':['x','y'],'values':xs,'covariance':Cs,'expression':'x+y'})
    assert r['status']=='PASS',r
