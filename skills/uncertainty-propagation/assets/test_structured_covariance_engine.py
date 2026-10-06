import sys
from pathlib import Path
import numpy as np, pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import structured_covariance_engine as s

def test_diagonal_low_rank_matches_dense_randomized():
    rng=np.random.default_rng(2)
    for _ in range(100):
        n=20;q=3;m=4;d=rng.uniform(.01,.5,n);U=rng.normal(size=(n,q));A=rng.normal(size=(q,q));K=A@A.T;J=rng.normal(size=(m,n))
        op=s.build({'kind':'diagonal_plus_low_rank','diagonal':d,'factor':U,'source_covariance':K})
        got=op.propagate(J);C=np.diag(d)+U@K@U.T
        assert np.allclose(got,J@C@J.T,rtol=1e-11,atol=1e-11)

def test_block_diagonal_matches_dense():
    rng=np.random.default_rng(3);blocks=[]
    for q in [3,5,2]:
        A=rng.normal(size=(q,q));blocks.append(A@A.T)
    J=rng.normal(size=(2,10));op=s.build({'kind':'block_diagonal','blocks':[b.tolist() for b in blocks]})
    from scipy.linalg import block_diag
    assert np.allclose(op.propagate(J),J@block_diag(*blocks)@J.T)

def test_sparse_factor_large_20000_without_dense_covariance():
    rng=np.random.default_rng(4);n=20000;q=4
    rows=np.arange(n).repeat(q);cols=np.tile(np.arange(q),n);data=rng.normal(scale=.01,size=n*q)
    op=s.build({'kind':'sparse_factor','diagonal':np.full(n,.2),'factor':{'shape':[n,q],'rows':rows,'cols':cols,'data':data}})
    g=rng.normal(size=n);v=op.quadratic_form(g)
    assert v>0 and np.isfinite(v); assert op.diagnostics['status']=='PSD_BY_CONSTRUCTION'
    with pytest.raises(s.InputError): op.to_dense()

def test_bad_source_covariance_rejected():
    with pytest.raises(s.InputError):s.build({'kind':'diagonal_plus_low_rank','diagonal':[1,1],'factor':[[1,0],[0,1]],'source_covariance':[[1,2],[2,1]]})

def test_arbitrary_large_sparse_fail_closed():
    with pytest.raises(s.InputError,match='not accepted'):
        s.build({'kind':'sparse_coo','shape':[3000,3000],'rows':[0,1],'cols':[0,1],'data':[1,1]})

def test_negative_diagonal_rejected():
    with pytest.raises(s.InputError):s.build({'kind':'diagonal','diagonal':[1,-1]})

def test_tiny_indefinite_dense_rejected():
    with pytest.raises(s.InputError):s.build({'kind':'dense','covariance':[[1e-40,2e-40],[2e-40,1e-40]]})

def test_tiny_material_asymmetry_dense_rejected():
    with pytest.raises(s.InputError):s.build({'kind':'dense','covariance':[[1e-40,8e-40],[0,1e-40]]})
