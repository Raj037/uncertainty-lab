import numpy as np
import pytest
import multifidelity_engine as m


def test_control_variate_reduces_standard_error_on_correlated_models():
    rng=np.random.default_rng(1);x=rng.normal(size=200);H=x+.2*rng.normal(size=200);Lp=x+.35*rng.normal(size=200);xe=rng.normal(size=5000);Le=xe+.35*rng.normal(size=5000)
    r=m.control_variate({'high_values':H.tolist(),'low_paired_values':Lp.tolist(),'low_extra_values':Le.tolist()})
    assert abs(r['estimate'])<.12
    assert r['standard_error_estimate'] < r['naive_high_only_standard_error']
    assert r['estimated_se_reduction_factor']>1.2


def test_control_variate_scale_invariance():
    rng=np.random.default_rng(2);x=rng.normal(size=100);H=2+x;L=2+.9*x;E=2+.9*rng.normal(size=3000)
    a=m.control_variate({'high_values':H,'low_paired_values':L,'low_extra_values':E})
    b=m.control_variate({'high_values':H*1e-20,'low_paired_values':L*1e-20,'low_extra_values':E*1e-20})
    assert b['estimate']/1e-20==pytest.approx(a['estimate'],rel=1e-12)
    assert b['estimated_se_reduction_factor']==pytest.approx(a['estimated_se_reduction_factor'],rel=1e-12)


def test_autoregressive_gp_qualifies_known_relation_and_propagates():
    rng=np.random.default_rng(4);Xl=np.linspace(-3,3,90)[:,None];yl=np.sin(Xl[:,0]);Xh=np.linspace(-2.8,2.8,28)[:,None];yh=1.2*np.sin(Xh[:,0])+.08*Xh[:,0]**2
    r=m.autoregressive_gp({'X_low':Xl.tolist(),'y_low':yl.tolist(),'X_high':Xh.tolist(),'y_high':yh.tolist(),'validation_fraction':.25,'validation_repeats':3,
        'qualification':{'max_nrmse':.08,'min_r2':.95},'noise_std':1e-5,
        'input_distribution':{'variables':['x'],'joint_model':{'kind':'multivariate_normal','mean':[0.0],'covariance':[[1.0]]}},'n':12000,'seed':7})
    assert r['status']=='SURROGATE_QUALIFIED',r['validation_metrics']
    assert r['propagation_status']=='PASS'
    assert abs(r['rho']-1.2)<.25


def test_bad_multifidelity_relation_is_rejected_by_high_fidelity_holdout():
    Xl=np.linspace(-2,2,50)[:,None];yl=Xl[:,0];Xh=np.linspace(-2,2,24)[:,None];yh=np.sin(8*Xh[:,0])
    r=m.autoregressive_gp({'X_low':Xl.tolist(),'y_low':yl.tolist(),'X_high':Xh.tolist(),'y_high':yh.tolist(),'validation_fraction':.3,'validation_repeats':3,
        'qualification':{'max_nrmse':.03,'min_r2':.98},'noise_std':1e-5})
    assert r['status']=='SURROGATE_REJECTED'
