import math, sys
from pathlib import Path
import numpy as np, pytest
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
import autodiff_engine as ad

def test_quadratic_exact_gradient_hessian():
    r=ad.differentiate('x*x + 3*x*y + y*y',['x','y'],[2.,-1.])
    assert r['values'][0] == pytest.approx(-1.)
    assert np.allclose(r['jacobian'][0],[1.,4.])
    assert np.allclose(r['hessians'][0],[[2.,3.],[3.,2.]])

def test_exp_log_sin_chain():
    x=.3
    r=ad.differentiate('exp(sin(x))+log(x+2)',['x'],[x])
    f1=math.exp(math.sin(x))*math.cos(x)+1/(x+2)
    f2=math.exp(math.sin(x))*(math.cos(x)**2-math.sin(x))-1/(x+2)**2
    assert r['jacobian'][0][0] == pytest.approx(f1,rel=1e-12)
    assert r['hessians'][0][0][0] == pytest.approx(f2,rel=1e-12)

def test_variable_exponent_positive_base():
    r=ad.differentiate('x**y',['x','y'],[2.,3.])
    assert r['values'][0]==pytest.approx(8.)
    assert np.allclose(r['jacobian'][0],[12.,8*math.log(2)],rtol=1e-12)

def test_domain_and_nondifferentiable_fail_closed():
    with pytest.raises(ad.InputError): ad.differentiate('log(x)',['x'],[-1.])
    with pytest.raises(ad.InputError): ad.differentiate('abs(x)',['x'],[0.])
    with pytest.raises(ad.InputError): ad.differentiate('__import__("os")',['x'],[1.])

def test_random_polynomials_against_closed_form():
    rng=np.random.default_rng(20261005)
    for _ in range(200):
        n=4; x=rng.normal(size=n); A=rng.normal(size=(n,n)); A=(A+A.T)/2; b=rng.normal(size=n)
        # Construct x^T A x+b^T x with explicit coefficients.
        terms=[]
        for i in range(n):
            terms.append(f'({b[i]:.17g})*x{i}')
            for j in range(n): terms.append(f'({A[i,j]:.17g})*x{i}*x{j}')
        ex='+'.join(terms); names=[f'x{i}' for i in range(n)]
        r=ad.differentiate(ex,names,x)
        want_g=(2*A@x+b); want_H=2*A
        assert np.allclose(r['jacobian'][0],want_g,rtol=2e-12,atol=2e-12)
        assert np.allclose(r['hessians'][0],want_H,rtol=2e-12,atol=2e-12)

def test_scale_extremes_linear_are_exact():
    for s in [1e-30,1e-20,1e20,1e30]:
        r=ad.differentiate('3*x-2*y',['x','y'],[s,-s])
        assert np.allclose(r['jacobian'][0],[3,-2],rtol=0,atol=0)
        assert np.allclose(r['hessians'][0],0,rtol=0,atol=0)
