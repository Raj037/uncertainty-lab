#!/usr/bin/env python3
"""Joint coverage regions for vector measurands."""
from __future__ import annotations
from typing import Any
import math
import numpy as np
from scipy import stats
ENGINE_VERSION='0.5.0'
MAX_DIMENSION=256
MAX_SAMPLES=2_000_000
class InputError(ValueError):pass

def _decomp(cov:Any):
    C=np.asarray(cov,dtype=float)
    if C.ndim!=2 or C.shape[0]!=C.shape[1] or C.shape[0]>MAX_DIMENSION or not np.isfinite(C).all():raise InputError(f'covariance must be finite square with dimension <= {MAX_DIMENSION}')
    d=np.diag(C)
    if np.any(d<0): raise InputError('covariance has negative variance')
    pos=d>0;zero=~pos
    if np.any(zero) and (np.any(C[zero,:]!=0) or np.any(C[:,zero]!=0)):raise InputError('zero-variance row has covariance')
    if np.any(pos):
        sd=np.sqrt(d[pos]);raw=C[np.ix_(pos,pos)]/np.outer(sd,sd)
        asym=float(np.max(np.abs(raw-raw.T)));scale=max(1.0,float(np.max(np.abs(raw))))
        if asym>1e-12+1e-10*scale:raise InputError('covariance not symmetric on variance-normalized basis')
        R=(raw+raw.T)/2;ev,Q=np.linalg.eigh(R);tol=1e-12+1e-10*max(1.0,float(np.max(np.abs(ev))))
        if ev[0]<-tol:raise InputError('covariance not PSD')
        keep=ev>tol
    else:
        sd=np.array([],dtype=float);R=np.zeros((0,0));ev=np.array([]);Q=np.zeros((0,0));keep=np.array([],dtype=bool)
    return (C+C.T)/2,pos,sd,Q[:,keep],ev[keep],int(np.sum(keep))

def gaussian_ellipsoid(mean:Any,covariance:Any,probability:float=0.95)->dict[str,Any]:
    mu=np.asarray(mean,dtype=float);C,pos,sd,B,lam,rank=_decomp(covariance)
    if mu.shape!=(C.shape[0],) or not np.isfinite(mu).all():raise InputError('mean mismatch')
    p=float(probability)
    if not 0<p<1:raise InputError('probability must be in (0,1)')
    radius_sq=0.0 if rank==0 else float(stats.chi2.ppf(p,df=rank))
    volume=None
    if rank==len(mu) and rank>0:
        unit=math.pi**(rank/2)/math.gamma(rank/2+1)
        # det(C)=prod(sd^2)*det(R), evaluated in logs for heterogeneous units.
        logdet=2*float(np.sum(np.log(sd)))+float(np.sum(np.log(lam)))
        volume=float(unit*(math.sqrt(radius_sq)**rank)*math.exp(0.5*logdet)) if abs(logdet)<1400 else None
    return {'engine_version':ENGINE_VERSION,'kind':'gaussian_ellipsoid','probability':p,'center':mu.tolist(),'covariance':C.tolist(),'rank':rank,
            'positive_variance_mask':pos.tolist(),'standard_deviations_positive':sd.tolist(),'normalized_basis':B.tolist(),'normalized_positive_eigenvalues':lam.tolist(),
            'mahalanobis_radius_squared':radius_sq,'full_dimensional_volume':volume,
            'interpretation':'Exact probability region only under the stated multivariate normal model.'}

def _mahalanobis(region:dict[str,Any],points:Any)->tuple[np.ndarray,np.ndarray]:
    X=np.asarray(points,dtype=float)
    if X.ndim==1:X=X.reshape(1,-1)
    mu=np.asarray(region['center']);D=X-mu;pos=np.asarray(region['positive_variance_mask'],dtype=bool)
    sd=np.asarray(region['standard_deviations_positive']);B=np.asarray(region['normalized_basis']);lam=np.asarray(region['normalized_positive_eigenvalues'])
    # Zero-variance axes define exact support constraints. With no physical scale
    # available, any nonzero displacement is outside support rather than being
    # excused by an arbitrary absolute-unit tolerance.
    if np.any(~pos): zero_res=np.where(np.any(D[:,~pos]!=0.0,axis=1),np.inf,0.0)
    else: zero_res=np.zeros(X.shape[0])
    if B.shape[1]==0:
        md=np.zeros(X.shape[0]);sub_res=np.max(np.abs(D[:,pos]),axis=1) if np.any(pos) else np.zeros(X.shape[0])
    else:
        Z=D[:,pos]/sd[None,:];proj=Z@B;md=np.sum((proj*proj)/lam[None,:],axis=1);resid=Z-proj@B.T;sub_res=np.linalg.norm(resid,axis=1)
    return md,np.maximum(zero_res,sub_res)

def contains(region:dict[str,Any],points:Any,subspace_atol:float=1e-10)->np.ndarray|bool:
    md,res=_mahalanobis(region,points);ok=(md<=float(region['mahalanobis_radius_squared'])*(1+1e-12))&(res<=subspace_atol)
    return bool(ok[0]) if np.asarray(points).ndim==1 else ok

def empirical_ellipsoid(samples:Any,probability:float=0.95)->dict[str,Any]:
    X=np.asarray(samples,dtype=float)
    if X.ndim!=2 or X.shape[0]<3 or X.shape[0]>MAX_SAMPLES or X.shape[1]>MAX_DIMENSION or not np.isfinite(X).all():raise InputError('samples must be finite bounded N x D')
    p=float(probability)
    if not 0<p<1:raise InputError('probability must be in (0,1)')
    mu=X.mean(axis=0);C=np.atleast_2d(np.cov(X,rowvar=False,ddof=1));base=gaussian_ellipsoid(mu,C,0.5)
    md,res=_mahalanobis(base,X)
    if np.any(~np.isfinite(res)) or np.max(res)>1e-8: raise InputError('sample points not represented by covariance support')
    # 'higher' makes in-sample coverage at least requested probability.
    q=float(np.quantile(md,p,method='higher'))
    base.update({'kind':'empirical_mahalanobis_region','probability':p,'mahalanobis_radius_squared':q,'sample_count':int(X.shape[0]),
                 'in_sample_coverage':float(np.mean(md<=q)),
                 'interpretation':'Descriptive empirical region calibrated to the supplied sample; not an exact future-sample confidence/coverage claim.'})
    return base

def empirical_maxnorm_box(samples:Any,probability:float=0.95)->dict[str,Any]:
    X=np.asarray(samples,dtype=float)
    if X.ndim!=2 or X.shape[0]<3 or X.shape[0]>MAX_SAMPLES or X.shape[1]>MAX_DIMENSION or not np.isfinite(X).all():raise InputError('samples must be finite bounded N x D')
    p=float(probability)
    if not 0<p<1:raise InputError('probability must be in (0,1)')
    center=np.median(X,axis=0);scale=np.median(np.abs(X-center),axis=0)*1.4826
    # fall back to sample SD for constant/MAD-zero axes; exact-zero axes remain zero.
    sd=X.std(axis=0,ddof=1);scale=np.where(scale>0,scale,sd)
    z=np.zeros_like(X);nz=scale>0;z[:,nz]=np.abs((X[:,nz]-center[nz])/scale[nz]);z[:,~nz]=np.where(X[:,~nz]==center[~nz],0,np.inf)
    score=np.max(z,axis=1);q=float(np.quantile(score,p,method='higher'));lo=center-q*scale;hi=center+q*scale
    return {'engine_version':ENGINE_VERSION,'kind':'empirical_simultaneous_box','probability':p,'center':center.tolist(),'scale':scale.tolist(),'threshold':q,
            'lower':lo.tolist(),'upper':hi.tolist(),'sample_count':int(X.shape[0]),'in_sample_coverage':float(np.mean(score<=q)),
            'interpretation':'Joint empirical box calibrated by max standardized deviation; descriptive for the supplied distribution/sample.'}
