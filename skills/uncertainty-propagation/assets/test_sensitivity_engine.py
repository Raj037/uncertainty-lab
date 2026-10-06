import sys,math
from pathlib import Path
import numpy as np,pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import sensitivity_engine as s

def test_sobol_linear_uniform_matches_exact_variance_shares():
    p={'variables':['x','y'],'marginals':[{'kind':'uniform','lower':-1,'upper':1},{'kind':'uniform','lower':-1,'upper':1}], 'expression':'x+2*y','n':180000,'replicates':3,'seed':1}
    r=s.sobol_independent(p);want=np.array([1,4])/5
    assert np.allclose(r['first_order_mean'],want,atol=.012);assert np.allclose(r['total_order_mean'],want,atol=.012)

def test_sobol_nonlinear_interaction_total_exceeds_first():
    p={'variables':['x','y'],'marginals':[{'kind':'uniform','lower':-1,'upper':1},{'kind':'uniform','lower':-1,'upper':1}], 'expression':'x*y','n':150000,'replicates':2,'seed':2}
    r=s.sobol_independent(p);assert all(t>f for f,t in zip(r['first_order_mean'],r['total_order_mean']))

def test_correlated_gaussian_shapley_linear_agrees_with_analytic_bivariate_game():
    rho=.6;a=1.;b=2.;V=a*a+b*b+2*a*b*rho
    v1=(a+b*rho)**2;v2=(b+a*rho)**2
    want=np.array([.5*v1+.5*(V-v2),.5*v2+.5*(V-v1)])/V
    p={'variables':['x','y'],'mean':[0,0],'covariance':[[1,rho],[rho,1]],'expression':'x+2*y','outer_samples':3000,'inner_samples':250,'replicates':2,'seed':3}
    r=s.shapley_correlated_gaussian(p)
    assert np.allclose(r['effects_mean'],want,atol=.035);assert abs(r['effects_sum']-1)<.04

def test_correlated_shapley_rejects_indefinite_covariance():
    with pytest.raises(s.InputError):s.shapley_correlated_gaussian({'variables':['x','y'],'mean':[0,0],'covariance':[[1,2],[2,1]],'expression':'x+y'})

def test_correlated_shapley_extreme_unit_rescaling_stable_for_equivalent_model():
    rho=.4
    base={'variables':['x','y'],'mean':[0,0],'covariance':[[1,rho],[rho,1]],'expression':'x+2*y','outer_samples':1800,'inner_samples':160,'replicates':1,'seed':21}
    a=s.shapley_correlated_gaussian(base)
    # x'=1e-20 x, y'=1e20 y; expression restores the same physical model.
    scaled={'variables':['xp','yp'],'mean':[0,0],'covariance':[[1e-40,rho],[rho,1e40]],'expression':'1e20*xp+2e-20*yp','outer_samples':1800,'inner_samples':160,'replicates':1,'seed':21}
    b=s.shapley_correlated_gaussian(scaled)
    assert np.allclose(a['effects_mean'],b['effects_mean'],atol=1e-10,rtol=1e-10)
