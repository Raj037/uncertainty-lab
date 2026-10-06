import math
import numpy as np
import pytest
from scipy import stats
import reliability_engine as r


def base(a=4.0):
    return {'variables':['x'],'mean':[0.0],'covariance':[[1.0]],'limit_state':f'{a}-x'}


def test_form_is_exact_for_linear_normal_limit_state():
    p=base(4.0); got=r.form(p); exact=float(stats.norm.cdf(-4.0))
    assert got['status']=='PASS'
    assert got['beta']==pytest.approx(4.0,abs=1e-8)
    assert got['failure_probability_FORM']==pytest.approx(exact,rel=1e-7)
    assert got['design_point_x'][0]==pytest.approx(4.0,abs=1e-8)


def test_sorm_reduces_to_form_for_linear_surface():
    p=base(3.0); got=r.sorm(p)
    assert got['status']=='PASS'
    assert got['principal_curvatures']==[]
    assert got['failure_probability_SORM_Breitung']==pytest.approx(got['failure_probability_FORM'],rel=1e-12)


def test_importance_sampling_resolves_four_sigma_tail():
    p={**base(4.0),'n':80000,'seed':42}
    got=r.importance_sampling(p); exact=float(stats.norm.cdf(-4.0))
    assert got['status']=='PASS'
    assert abs(got['failure_probability']-exact) < 5*got['standard_error']
    assert got['coefficient_of_variation'] < 0.03


def test_subset_simulation_tracks_known_tail():
    p={**base(3.5),'n':3000,'p0':0.1,'replicates':4,'seed':123,'proposal_scale':0.8}
    got=r.subset_simulation(p); exact=float(stats.norm.cdf(-3.5))
    assert got['status'] in {'PASS','PROVISIONAL'}
    # stochastic rare-event method: require factor-of-two agreement and stable replicate scale
    ratio=got['failure_probability_mean']/exact
    assert 0.5 < ratio < 2.0


def test_two_dimensional_linear_form_matches_closed_form():
    # g=5-(x+2y), independent standard normals => beta=5/sqrt(5)
    p={'variables':['x','y'],'mean':[0,0],'covariance':[[1,0],[0,1]],'limit_state':'5-x-2*y'}
    got=r.form(p); beta=5/math.sqrt(5)
    assert got['beta']==pytest.approx(beta,rel=1e-8)
    assert got['failure_probability_FORM']==pytest.approx(float(stats.norm.cdf(-beta)),rel=1e-7)
    assert sum(got['importance_factors_u_squared'])==pytest.approx(1.0,abs=1e-10)


def test_singular_covariance_fails_closed():
    with pytest.raises(r.InputError):
        r.form({'variables':['x','y'],'mean':[0,0],'covariance':[[1,1],[1,1]],'limit_state':'3-x-y'})


def test_sorm_improves_over_form_on_mild_curvature_against_quadrature():
    from scipy import integrate
    q=.05
    p={'variables':['x','y'],'mean':[0,0],'covariance':[[1,0],[0,1]],'limit_state':f'3-x-{q}*y**2'}
    f=r.form(p);s=r.sorm(p)
    exact=integrate.quad(lambda yy: stats.norm.pdf(yy)*stats.norm.sf(3-q*yy*yy),-10,10,epsabs=1e-13,epsrel=1e-11)[0]
    assert s['status']=='PASS'
    assert abs(s['failure_probability_SORM_Breitung']-exact) < abs(f['failure_probability_FORM']-exact)


def test_sorm_blocks_when_breitung_curvature_factor_is_invalid():
    p={'variables':['x','y'],'mean':[0,0],'covariance':[[1,0],[0,1]],'limit_state':'3-x-0.2*y**2'}
    s=r.sorm(p)
    assert s['status']=='BLOCKED'
