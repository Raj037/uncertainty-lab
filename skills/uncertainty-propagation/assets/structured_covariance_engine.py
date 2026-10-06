#!/usr/bin/env python3
"""Structured/sparse covariance propagation without mandatory dense materialization."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import math
import numpy as np
from scipy import sparse

ENGINE_VERSION="0.5.0"
MAX_DENSE_AUDIT=2048
MAX_OUTPUTS=256
class InputError(ValueError):pass

def _psd_dense(a:Any,name='matrix')->tuple[np.ndarray,dict[str,Any]]:
    x=np.asarray(a,dtype=float)
    if x.ndim!=2 or x.shape[0]!=x.shape[1] or not np.isfinite(x).all():raise InputError(f'{name} must be finite square')
    d=np.diag(x)
    if np.any(d<0):raise InputError(f'{name} has negative diagonal')
    pos=d>0
    if np.any(~pos) and (np.any(x[~pos,:]!=0) or np.any(x[:,~pos]!=0)):raise InputError(f'{name}: zero variance row has nonzero covariance')
    if np.any(pos):
        sd0=np.sqrt(d[pos]);Rn=x[np.ix_(pos,pos)]/np.outer(sd0,sd0);scale=max(1.0,float(np.max(np.abs(Rn))))
        if np.max(np.abs(Rn-Rn.T))>1e-12+1e-10*scale:raise InputError(f'{name} not symmetric on variance-normalized basis')
    s=(x+x.T)/2;d=np.diag(s);pos=d>0
    if np.any(pos):
        sd=np.sqrt(d[pos]);r=s[np.ix_(pos,pos)]/np.outer(sd,sd);ev=np.linalg.eigvalsh((r+r.T)/2);tol=1e-12+1e-10*max(1.0,float(np.max(np.abs(ev))))
        if ev[0]<-tol:raise InputError(f'{name} not PSD')
        rank=int(np.sum(ev>tol));cond=None if rank<=1 else float(ev[-1]/ev[ev>tol][0])
    else:ev=np.array([]);rank=0;cond=None
    return s,{"numerical_rank":rank,"condition_normalized":cond,"min_normalized_eigenvalue":float(ev[0]) if ev.size else 0.0}

def _factor_from_source_cov(k:np.ndarray)->np.ndarray:
    k,diag=_psd_dense(k,'source_covariance');ev,Q=np.linalg.eigh(k)
    if not ev.size:return np.zeros((k.shape[0],0))
    scale=float(np.max(np.abs(ev)))
    tol=(1e-12+1e-10)*scale if scale>0 else 0.0
    keep=ev>tol
    return Q[:,keep]*np.sqrt(ev[keep])

def _coo(m:Mapping[str,Any],name='factor')->sparse.csr_matrix:
    shape=tuple(m.get('shape') or ());rows=np.asarray(m.get('rows'),dtype=int);cols=np.asarray(m.get('cols'),dtype=int);data=np.asarray(m.get('data'),dtype=float)
    if len(shape)!=2 or min(shape)<0 or rows.shape!=cols.shape or rows.shape!=data.shape or not np.isfinite(data).all():raise InputError(f'invalid {name} COO')
    if rows.size and (rows.min()<0 or cols.min()<0 or rows.max()>=shape[0] or cols.max()>=shape[1]):raise InputError(f'{name} index out of bounds')
    return sparse.coo_matrix((data,(rows,cols)),shape=shape).tocsr()

@dataclass
class CovarianceOperator:
    kind:str
    n:int
    payload:dict[str,Any]
    diagnostics:dict[str,Any]

    def propagate(self,J:Any)->np.ndarray:
        J=np.asarray(J,dtype=float)
        if J.ndim==1:J=J.reshape(1,-1)
        if J.ndim!=2 or J.shape[0]>MAX_OUTPUTS or J.shape[1]!=self.n or not np.isfinite(J).all():raise InputError(f'linear_map must be finite with <= {MAX_OUTPUTS} outputs and one column per input')
        p=self.payload
        if self.kind=='dense': out=J@p['cov']@J.T
        elif self.kind=='diagonal': out=(J*p['diag'][None,:])@J.T
        elif self.kind=='diagonal_plus_low_rank':
            B=J@p['U'];out=(J*p['diag'][None,:])@J.T+B@p['K']@B.T
        elif self.kind=='sparse_factor':
            B=np.asarray(J@p['L']);out=(J*p['diag'][None,:])@J.T+B@p['K']@B.T
        elif self.kind=='block_diagonal':
            out=np.zeros((J.shape[0],J.shape[0]));offset=0
            for C in p['blocks']:
                q=C.shape[0];B=J[:,offset:offset+q];out+=B@C@B.T;offset+=q
        elif self.kind=='sparse_coo_dense_audited':out=J@p['cov']@J.T
        else:raise InputError('unknown operator')
        return (out+out.T)/2

    def quadratic_form(self,g:Any)->float:
        g=np.asarray(g,dtype=float)
        if g.shape!=(self.n,):raise InputError('g shape mismatch')
        return float(self.propagate(g)[0,0])

    def to_dense(self,max_dimension:int=MAX_DENSE_AUDIT)->np.ndarray:
        if self.n>max_dimension:raise InputError(f'dense materialization blocked above dimension {max_dimension}')
        I=np.eye(self.n);return self.propagate(I)

def build(spec:Mapping[str,Any])->CovarianceOperator:
    s=dict(spec);kind=str(s.get('kind','')).lower()
    if kind=='dense':
        C,d=_psd_dense(s.get('covariance'));return CovarianceOperator(kind,C.shape[0],{'cov':C},{'status':'PSD_VALIDATED_DENSE',**d})
    if kind=='diagonal':
        d=np.asarray(s.get('diagonal'),dtype=float)
        if d.ndim!=1 or not np.isfinite(d).all() or np.any(d<0):raise InputError('diagonal must be finite nonnegative vector')
        rank=int(np.sum(d>0));return CovarianceOperator(kind,len(d),{'diag':d},{'status':'PSD_BY_CONSTRUCTION','numerical_rank':rank})
    if kind=='diagonal_plus_low_rank':
        d=np.asarray(s.get('diagonal'),dtype=float);U=np.asarray(s.get('factor'),dtype=float)
        if d.ndim!=1 or U.ndim!=2 or U.shape[0]!=len(d) or not np.isfinite(d).all() or not np.isfinite(U).all() or np.any(d<0):raise InputError('invalid diagonal/factor')
        K=np.eye(U.shape[1]) if s.get('source_covariance') is None else _psd_dense(s['source_covariance'],'source_covariance')[0]
        if K.shape!=(U.shape[1],U.shape[1]):raise InputError('source_covariance dimension mismatch')
        rank=len(d) if np.all(d>0) else (int(np.linalg.matrix_rank(np.c_[np.diag(np.sqrt(d)),U@_factor_from_source_cov(K)])) if len(d)<=MAX_DENSE_AUDIT else None)
        return CovarianceOperator(kind,len(d),{'diag':d,'U':U,'K':K},{'status':'PSD_BY_CONSTRUCTION','numerical_rank':rank,'rank_basis':'factorization' if rank is not None else 'not_computed_large_structured'})
    if kind=='sparse_factor':
        d=np.asarray(s.get('diagonal'),dtype=float);L=_coo(s.get('factor') or {},'factor')
        if d.ndim!=1 or len(d)!=L.shape[0] or not np.isfinite(d).all() or np.any(d<0):raise InputError('invalid sparse factor diagonal')
        K=np.eye(L.shape[1]) if s.get('source_covariance') is None else _psd_dense(s['source_covariance'],'source_covariance')[0]
        if K.shape!=(L.shape[1],L.shape[1]):raise InputError('source covariance dimension mismatch')
        return CovarianceOperator(kind,len(d),{'diag':d,'L':L,'K':K},{'status':'PSD_BY_CONSTRUCTION','nnz_factor':int(L.nnz),'numerical_rank':len(d) if np.all(d>0) else None})
    if kind=='block_diagonal':
        blocks=[];rank=0
        for i,b in enumerate(s.get('blocks') or []):
            C,d=_psd_dense(b,f'block[{i}]');blocks.append(C);rank+=d['numerical_rank']
        if not blocks:raise InputError('blocks required')
        n=sum(x.shape[0] for x in blocks);return CovarianceOperator(kind,n,{'blocks':blocks},{'status':'PSD_VALIDATED_BY_BLOCKS','numerical_rank':rank,'block_count':len(blocks)})
    if kind=='sparse_coo':
        C=_coo(s,'covariance')
        if C.shape[0]!=C.shape[1]:raise InputError('sparse covariance must be square')
        if C.shape[0]>MAX_DENSE_AUDIT:raise InputError('arbitrary sparse covariance above dense-audit threshold is not accepted; provide sparse_factor/diagonal_plus_low_rank so PSD is proven by construction')
        dense=C.toarray();dense,d=_psd_dense(dense,'sparse covariance')
        return CovarianceOperator('sparse_coo_dense_audited',C.shape[0],{'cov':dense},{'status':'PSD_VALIDATED_AFTER_BOUNDED_DENSE_AUDIT',**d,'nnz':int(C.nnz)})
    raise InputError('unsupported structured covariance kind')

def linear_propagate(spec:Mapping[str,Any],linear_map:Any)->dict[str,Any]:
    op=build(spec);C=op.propagate(linear_map);sd=np.sqrt(np.clip(np.diag(C),0,None))
    return {'engine_version':ENGINE_VERSION,'input_dimension':op.n,'output_dimension':C.shape[0],'representation':op.kind,'validation':op.diagnostics,'output_covariance':C.tolist(),'output_std_uncertainty':sd.tolist()}
