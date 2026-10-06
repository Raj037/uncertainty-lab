import sys
from pathlib import Path
import numpy as np, pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import coverage_regions as c

def test_gaussian_ellipsoid_has_correct_simulated_coverage():
    rng=np.random.default_rng(5);C=np.array([[2,.7],[.7,1.]])
    r=c.gaussian_ellipsoid([1,-2],C,.95);X=rng.multivariate_normal([1,-2],C,size=200000)
    frac=np.mean(c.contains(r,X));assert abs(frac-.95)<.003

def test_affine_rescaling_preserves_membership():
    rng=np.random.default_rng(6);C=np.array([[1,.3],[.3,2.]]) ; mu=np.array([.5,-1.]);r=c.gaussian_ellipsoid(mu,C,.9);X=rng.normal(size=(100,2))
    D=np.diag([1e-12,1e9]);r2=c.gaussian_ellipsoid(D@mu,D@C@D,.9)
    assert np.array_equal(c.contains(r,X),c.contains(r2,X@D.T,subspace_atol=1e-6))

def test_singular_region_uses_rank_subspace():
    C=np.array([[1,1],[1,1]],float);r=c.gaussian_ellipsoid([0,0],C,.95);assert r['rank']==1
    assert c.contains(r,[0.5,0.5])
    assert not c.contains(r,[0.5,0.6])

def test_empirical_region_meets_requested_in_sample_coverage():
    rng=np.random.default_rng(7);X=np.c_[rng.standard_t(4,5000),rng.gamma(2,size=5000)]
    r=c.empirical_ellipsoid(X,.9);assert r['in_sample_coverage']>=.9;assert r['kind']=='empirical_mahalanobis_region'

def test_empirical_box_joint_coverage_calibrated():
    rng=np.random.default_rng(8);X=rng.normal(size=(5000,4));r=c.empirical_maxnorm_box(X,.95);assert r['in_sample_coverage']>=.95

def test_bad_covariance_rejected():
    with pytest.raises(c.InputError):c.gaussian_ellipsoid([0,0],[[1,2],[2,1]],.95)

def test_zero_variance_axis_is_exact_support_constraint():
    r=c.gaussian_ellipsoid([0.0,5.0],[[1.0,0.0],[0.0,0.0]],0.95)
    assert c.contains(r,[0.0,5.0]) is True
    assert c.contains(r,[0.0,np.nextafter(5.0,np.inf)]) is False
