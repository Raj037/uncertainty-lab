import numpy as np
import pytest
import dependent_attribution_engine as dep


def test_independent_uniform_additive_recovers_about_20_80():
    p={
      'variables':['x','y'],
      'joint_model':{'kind':'gaussian_copula','copula_correlation':[[1,0],[0,1]],
                     'marginals':[{'kind':'uniform','lower':-1,'upper':1},{'kind':'uniform','lower':-1,'upper':1}]},
      'n':12000,'seed':11,'attribution_seed':29,'expression':'x+2*y','replicates':4,'k_neighbors':55,
      'max_effect_standard_error':.04,
    }
    r=dep.shapley_empirical(p)
    assert r['quality']=='STABLE'
    assert r['effects_sum']==pytest.approx(1.0,abs=1e-10)
    assert r['effects_mean'][0]==pytest.approx(.2,abs=.07)
    assert r['effects_mean'][1]==pytest.approx(.8,abs=.07)


def test_correlated_nongaussian_t_copula_symmetric_effects():
    p={
      'variables':['a','b'],
      'joint_model':{'kind':'t_copula','df':5,'copula_correlation':[[1,.65],[.65,1]],
                     'marginals':[{'kind':'beta','a':2,'b':5},{'kind':'beta','a':2,'b':5}]},
      'n':15000,'seed':37,'attribution_seed':101,'expression':'a+b','replicates':4,'k_neighbors':60,
      'max_effect_standard_error':.05,
    }
    r=dep.shapley_empirical(p)
    assert r['joint_model']['kind']=='t_copula'
    assert r['quality']=='STABLE'
    assert r['effects_sum']==pytest.approx(1.0,abs=1e-10)
    assert r['effects_mean'][0]==pytest.approx(.5,abs=.08)
    assert r['effects_mean'][1]==pytest.approx(.5,abs=.08)


def test_provided_empirical_samples_are_deterministic():
    rng=np.random.default_rng(5)
    x=rng.normal(size=6000); y=.7*x+rng.normal(size=6000)
    samples=np.column_stack([x,y])
    p={'variables':['x','y'],'joint_model':{'kind':'empirical_samples'},'samples':samples.tolist(),
       'expression':'x*y','replicates':3,'attribution_seed':7,'k_neighbors':40}
    a=dep.shapley_empirical(p); b=dep.shapley_empirical(p)
    assert a==b


def test_fail_closed_dimension_and_sample_limits():
    with pytest.raises(dep.InputError):
        dep.shapley_empirical({'variables':['x']*9,'joint_model':{'kind':'empirical_samples'},'samples':[[0]*9]*1000,'expression':'x'})
    with pytest.raises(dep.InputError):
        dep.shapley_empirical({'variables':['x'],'joint_model':{'kind':'empirical_samples'},'samples':[[float(i)] for i in range(20)],'expression':'x'})

def test_empirical_estimator_matches_exact_linear_gaussian_shapley():
    # X~N(0, [[1,.5],[.5,1]]), Y=X1+2X2.
    # For d=2, v({i})=Cov(Y,Xi)^2/Var(Xi), V=7, giving
    # phi1=.5*(4 + (7-6.25))/7 = 0.339285714..., phi2=1-phi1.
    p={
      'variables':['x1','x2'],
      'joint_model':{'kind':'multivariate_normal','mean':[0,0],'covariance':[[1,.5],[.5,1]]},
      'n':18000,'seed':222,'attribution_seed':333,'expression':'x1+2*x2',
      'replicates':5,'k_neighbors':65,'max_effect_standard_error':.04,
    }
    r=dep.shapley_empirical(p)
    exact1=2.375/7.0
    assert r['quality']=='STABLE'
    assert r['effects_mean'][0]==pytest.approx(exact1,abs=.06)
    assert r['effects_mean'][1]==pytest.approx(1-exact1,abs=.06)
