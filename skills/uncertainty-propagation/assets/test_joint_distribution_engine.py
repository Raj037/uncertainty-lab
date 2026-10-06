import math,sys
from pathlib import Path
import numpy as np, pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import joint_distribution_engine as j

def test_empirical_joint_samples_preserve_dependence():
    rng=np.random.default_rng(1);x=rng.normal(size=20000);y=.8*x+rng.normal(scale=.6,size=20000);X=np.c_[x,y]
    r=j.propagate({'variables':['x','y'],'joint_model':{'kind':'empirical_samples'},'samples':X.tolist(),'expressions':['x+y','x-y']})
    want=np.cov(np.c_[x+y,x-y],rowvar=False,ddof=1)
    assert np.allclose(r['output_covariance'],want,rtol=1e-12,atol=1e-12)

def test_gaussian_copula_uniform_reproducible_and_correlated():
    p={'variables':['x','y'],'joint_model':{'kind':'gaussian_copula','copula_correlation':[[1,.7],[.7,1]],'marginals':[{'kind':'uniform','lower':-1,'upper':1},{'kind':'uniform','lower':-1,'upper':1}]},'n':120000,'seed':123,'expression':'x+y'}
    a=j.propagate(p);b=j.propagate(p)
    assert a==b
    c=np.array(a['input_sample_covariance']); assert c[0,1]>0.18
    assert abs(a['mean'][0])<.01

def test_t_copula_has_more_joint_tail_than_gaussian_same_latent_corr():
    base={'variables':['x','y'],'n':180000,'seed':42,'marginals':[{'kind':'normal'},{'kind':'normal'}]}
    pg={**base,'joint_model':{'kind':'gaussian_copula','copula_correlation':[[1,.5],[.5,1]],'marginals':base['marginals']}}
    pt={**base,'joint_model':{'kind':'t_copula','df':3,'copula_correlation':[[1,.5],[.5,1]],'marginals':base['marginals']}}
    _,Xg,_=j.generate_inputs(pg);_,Xt,_=j.generate_inputs(pt)
    # Probability both transformed normals exceed 2 sigma is larger under low-df t copula.
    assert np.mean((Xt[:,0]>2)&(Xt[:,1]>2)) > np.mean((Xg[:,0]>2)&(Xg[:,1]>2))*1.3

def test_invalid_copula_is_rejected():
    p={'variables':['x','y'],'joint_model':{'kind':'gaussian_copula','copula_correlation':[[1,2],[2,1]],'marginals':[{'kind':'normal'},{'kind':'normal'}]},'n':1000}
    with pytest.raises(j.InputError): j.generate_inputs(p)

def test_correlation_alone_does_not_define_nongaussian_joint():
    with pytest.raises(j.InputError): j.generate_inputs({'variables':['x'],'joint_model':{},'n':1000})

def test_domain_violations_are_explicit_not_dropped_silently():
    p={'variables':['x'],'joint_model':{'kind':'gaussian_copula','copula_correlation':[[1]],'marginals':[{'kind':'normal','mean':0,'std':1}]},'n':20000,'seed':9,'expression':'log(x)'}
    r=j.propagate(p);assert r['status']=='domain_violations';assert .45<r['invalid_fraction']<.55;assert r['conditional_on_valid'] is True

def test_multivariate_t_reproducible():
    p={'variables':['x','y'],'joint_model':{'kind':'multivariate_t','mean':[0,0],'scale_matrix':[[1,.2],[.2,1]],'df':5},'n':50000,'seed':7,'expression':'x*y'}
    assert j.propagate(p)==j.propagate(p)

def test_invalid_normal_marginal_scale_blocked():
    p={'variables':['x'],'joint_model':{'kind':'gaussian_copula','copula_correlation':[[1.0]],'marginals':[{'kind':'normal','mean':0,'std':0}]},'n':1000,'expression':'x'}
    with pytest.raises(j.InputError,match='positive standard deviation'):
        j.propagate(p)

def test_scalar_input_sample_covariance_is_matrix():
    p={'variables':['x'],'joint_model':{'kind':'multivariate_normal','mean':[0],'covariance':[[2.0]]},'n':1000,'seed':5,'expression':'x'}
    r=j.propagate(p)
    assert np.asarray(r['input_sample_covariance']).shape==(1,1)
    assert np.asarray(r['output_covariance']).shape==(1,1)
