import sys,math
from pathlib import Path
import numpy as np,pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import timeseries_engine as t

def test_iid_matches_s_over_sqrt_n():
    x=np.array([1.,2.,3.,4.]);r=t.iid_mean(x);assert r['standard_uncertainty']==pytest.approx(x.std(ddof=1)/2)

def test_ar1_finite_n_formula_known_phi():
    rng=np.random.default_rng(10);phi=.8;n=4000;e=rng.normal(size=n);x=np.zeros(n)
    for i in range(1,n):x[i]=phi*x[i-1]+e[i]
    ri=t.iid_mean(x);ra=t.ar1_mean(x,phi=phi)
    assert ra['standard_uncertainty']>ri['standard_uncertainty']*2
    assert ra['phi']==phi

def test_estimated_ar1_close_on_long_series():
    rng=np.random.default_rng(11);phi=.55;n=8000;e=rng.normal(size=n);x=np.zeros(n)
    for i in range(1,n):x[i]=phi*x[i-1]+e[i]
    r=t.ar1_mean(x,estimate_phi=True);assert abs(r['phi']-phi)<.03

def test_hac_positive_and_explicit_bandwidth():
    rng=np.random.default_rng(12);x=np.cumsum(rng.normal(size=500))*0.02+rng.normal(size=500)
    r=t.hac_mean(x,bandwidth=8);assert r['bandwidth']==8 and r['standard_uncertainty']>=0

def test_block_bootstrap_is_deterministic():
    x=np.sin(np.arange(100)/8)+np.arange(100)*.001
    a=t.circular_block_bootstrap_mean(x,block_length=10,resamples=1000,seed=9);b=t.circular_block_bootstrap_mean(x,block_length=10,resamples=1000,seed=9);assert a==b

def test_balanced_random_effects_recovers_components_approximately():
    rng=np.random.default_rng(13);g=100;m=20;tau=.7;sig=.4;u=rng.normal(scale=tau,size=g);G=3+u[:,None]+rng.normal(scale=sig,size=(g,m));r=t.balanced_random_effects(G)
    assert abs(math.sqrt(r['between_group_variance'])-tau)<.12;assert abs(math.sqrt(r['within_group_variance'])-sig)<.04

def test_no_auto_model_selection():
    with pytest.raises(t.InputError):t.analyze({'observations':[1,2,3,4]})

def test_hac_is_scale_equivariant_at_tiny_physical_scale():
    x=np.array([1.0,-.5,.8,-.2,.3,-.1,.4,-.7,1.1,-.6]*20)
    a=t.hac_mean(x,bandwidth=4)
    b=t.hac_mean(x*1e-20,bandwidth=4)
    assert b['standard_uncertainty']==pytest.approx(a['standard_uncertainty']*1e-20,rel=1e-12,abs=0)
