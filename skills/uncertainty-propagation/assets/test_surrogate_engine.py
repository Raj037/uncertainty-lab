import math
import numpy as np
import pytest
import surrogate_engine as s


def test_pce_exact_quadratic_and_moments():
    rng=np.random.default_rng(1);X=rng.uniform(-1,1,size=(180,2));y=1+2*X[:,0]+3*X[:,1]**2
    r=s.polynomial_chaos({'X':X.tolist(),'y':y.tolist(),'marginals':[{'kind':'uniform','lower':-1,'upper':1}]*2,
        'total_degree':2,'validation_fraction':.25,'validation_seed':11,'qualification':{'max_nrmse':1e-10,'min_r2':.999999999}})
    assert r['status']=='SURROGATE_QUALIFIED'
    assert r['validation_metrics']['rmse'] < 1e-10
    assert r['distribution_mean_from_coefficients']==pytest.approx(2.0,abs=1e-10)
    assert r['distribution_variance_from_coefficients']==pytest.approx(4/3+0.8,abs=1e-9)


def test_pce_unqualified_without_thresholds():
    x=np.linspace(-1,1,30)[:,None];y=x[:,0]**2
    r=s.polynomial_chaos({'X':x.tolist(),'y':y.tolist(),'marginals':[{'kind':'uniform','lower':-1,'upper':1}],'total_degree':2})
    assert r['status']=='PROVISIONAL'


def test_gp_sine_holdout_is_accurate():
    x=np.linspace(-3,3,90)[:,None];y=np.sin(x[:,0])
    r=s.gaussian_process({'X':x.tolist(),'y':y.tolist(),'validation_fraction':.2,'validation_seed':7,'noise_std':1e-6,
                          'qualification':{'max_nrmse':.02,'min_r2':.99}})
    assert r['status']=='SURROGATE_QUALIFIED'
    assert r['validation_metrics']['r2'] > .99


def test_gp_propagation_separates_input_and_surrogate_variance():
    x=np.linspace(-3,3,100)[:,None];y=2*x[:,0]
    p={'X':x.tolist(),'y':y.tolist(),'validation_fraction':.2,'validation_seed':3,'noise_std':1e-6,'qualification':{'max_nrmse':.02,'min_r2':.999},
       'input_distribution':{'variables':['x'],'joint_model':{'kind':'multivariate_normal','mean':[0.0],'covariance':[[1.0]]}},'n':30000,'seed':9}
    r=s.propagate_gp(p)
    assert r['propagation_status']=='PASS'
    assert r['variance_due_to_input_through_surrogate_mean']==pytest.approx(4.0,rel=.04)
    assert r['mean_GP_predictive_variance'] < .01


def test_pce_propagation_uses_orthonormal_coefficients():
    rng=np.random.default_rng(8);x=rng.normal(size=(160,1));y=1+3*x[:,0]
    r=s.propagate_pce({'X':x.tolist(),'y':y.tolist(),'marginals':[{'kind':'normal','mean':0,'std':1}],'total_degree':1,
        'validation_fraction':.2,'qualification':{'max_nrmse':1e-10,'min_r2':.999999999}})
    assert r['propagation_status']=='PASS'
    assert r['output_mean']==pytest.approx(1.0,abs=1e-10)
    assert r['output_variance']==pytest.approx(9.0,abs=1e-9)


def test_gp_rejects_constant_input_axis():
    X=np.column_stack([np.arange(20),np.ones(20)]);y=X[:,0]
    with pytest.raises(s.InputError):
        s.gaussian_process({'X':X.tolist(),'y':y.tolist(),'qualification':{'min_r2':.9}})
