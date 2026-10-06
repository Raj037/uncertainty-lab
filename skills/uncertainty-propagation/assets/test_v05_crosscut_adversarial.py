import sys
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import joint_distribution_engine as joint
import structured_covariance_engine as structured
import coverage_regions as coverage
import v05_engine as v05


def test_joint_multivariate_normal_tiny_indefinite_rejected():
    p={'variables':['x','y'],'joint_model':{'kind':'multivariate_normal','mean':[0,0],
       'covariance':[[1e-40,2e-40],[2e-40,1e-40]]},'n':1000,'expression':'x+y'}
    with pytest.raises(joint.InputError,match='not PSD'):
        joint.propagate(p)


def test_joint_multivariate_t_tiny_indefinite_rejected():
    p={'variables':['x','y'],'joint_model':{'kind':'multivariate_t','mean':[0,0],
       'scale_matrix':[[1e-40,2e-40],[2e-40,1e-40]],'df':5},'n':1000,'expression':'x+y'}
    with pytest.raises(joint.InputError,match='not PSD'):
        joint.propagate(p)


def test_joint_tiny_valid_covariance_not_collapsed_to_zero():
    p={'variables':['x','y'],'joint_model':{'kind':'multivariate_normal','mean':[0,0],
       'covariance':[[1e-40,0],[0,2e-40]]},'n':20000,'seed':19,'expression':'x+y'}
    r=joint.propagate(p)
    assert r['std'][0] > 1e-21
    assert r['std'][0] == pytest.approx(np.sqrt(3e-40),rel=.03)


def test_structured_tiny_source_covariance_preserves_rank_and_propagation():
    spec={'kind':'diagonal_plus_low_rank','diagonal':[0.0,0.0],
          'factor':[[1.0,0.0],[0.0,1.0]],
          'source_covariance':[[1e-40,0],[0,2e-40]]}
    op=structured.build(spec)
    assert op.diagnostics['numerical_rank']==2
    got=op.propagate(np.eye(2))
    assert np.allclose(got,np.diag([1e-40,2e-40]),rtol=1e-14,atol=0)


def test_structured_output_dimension_ceiling():
    spec={'kind':'diagonal','diagonal':[1.0]}
    op=structured.build(spec)
    with pytest.raises(structured.InputError,match='outputs'):
        op.propagate(np.ones((structured.MAX_OUTPUTS+1,1)))


def test_joint_empirical_dimension_ceiling():
    p={'joint_model':{'kind':'empirical_samples'},'samples':np.zeros((2,joint.MAX_VARIABLES+1)).tolist(),'expression':'x0'}
    with pytest.raises(joint.InputError,match='bounded'):
        joint.propagate(p)


def test_joint_expression_count_ceiling():
    p={'variables':['x'],'joint_model':{'kind':'multivariate_normal','mean':[0], 'covariance':[[1]]},
       'n':100,'expressions':['x']*(joint.MAX_EXPRESSIONS+1)}
    with pytest.raises(joint.InputError,match='expression count'):
        joint.propagate(p)


def test_coverage_dimension_ceiling():
    n=coverage.MAX_DIMENSION+1
    with pytest.raises(coverage.InputError,match='dimension'):
        coverage.gaussian_ellipsoid(np.zeros(n),np.eye(n))


def test_unknown_operation_fails_closed():
    with pytest.raises(v05.InputError,match='unknown'):
        v05.run_request({'operation':'invented_automatic_magic'})
