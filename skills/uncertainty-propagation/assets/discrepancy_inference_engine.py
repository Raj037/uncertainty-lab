#!/usr/bin/env python3
"""Joint parameter / additive model-discrepancy inference for v0.8.

Qualified route: linear-Gaussian calibration with discrepancy covariance tau^2 K,
normal parameter prior, and an explicit half-normal prior on tau. Tau is integrated
on a log grid; parameter posteriors conditional on tau are analytic and then mixed.
An identifiability diagnostic measures how much discrepancy-kernel variance lies in
the model's parameter-effect column space. High confounding remains provisional.
"""
from __future__ import annotations
import math
from typing import Any, Mapping
import numpy as np
from scipy import linalg, special
import calibration_engine as cal
import joint_distribution_engine as joint

ENGINE_VERSION='0.8.0'
class InputError(ValueError): pass

def _psd_k(K,n):
    K=np.asarray(K,dtype=float)
    if K.shape!=(n,n):raise InputError('discrepancy_kernel dimension mismatch')
    try:K=joint.validate_covariance(K,'discrepancy_kernel')
    except Exception as exc:raise InputError(str(exc)) from exc
    tr=float(np.trace(K))
    if tr<=0:raise InputError('discrepancy_kernel must have positive trace')
    return K

def _confounding(A,K):
    # Fraction of discrepancy variance represented inside the column space of A.
    Q,_=np.linalg.qr(np.asarray(A,dtype=float),mode='reduced')
    if Q.size==0:return 0.0
    num=float(np.trace(Q.T@K@Q));den=float(np.trace(K));return min(1.0,max(0.0,num/den))

def infer_linear_gaussian(payload:Mapping[str,Any])->dict[str,Any]:
    y=np.asarray(payload.get('observed'),dtype=float);A=np.asarray(payload.get('design_matrix'),dtype=float)
    if y.ndim!=1 or y.size<3 or not np.isfinite(y).all():raise InputError('observed must be finite vector with >=3 values')
    n=y.size
    if A.ndim!=2 or A.shape[0]!=n or A.shape[1]<1 or not np.isfinite(A).all():raise InputError('design_matrix mismatch')
    d=A.shape[1];off=np.asarray(payload.get('offset',np.zeros(n)),dtype=float)
    if off.ndim==0:off=np.full(n,float(off))
    if off.shape!=(n,) or not np.isfinite(off).all():raise InputError('offset mismatch')
    m0=np.asarray(payload.get('prior_mean'),dtype=float);P0=np.asarray(payload.get('prior_covariance'),dtype=float)
    if m0.shape!=(d,):raise InputError('prior_mean mismatch')
    P0=cal._spd(P0,d,'prior_covariance')
    # Observation covariance only: discrepancy is inferred here, not supplied to _obs_cov.
    p0=dict(payload);p0.pop('discrepancy_covariance',None)
    R,_,_=cal._obs_cov(p0,n)
    K=_psd_k(payload.get('discrepancy_kernel',np.eye(n)),n)
    frac=_confounding(A,K);threshold=float(payload.get('max_confounding_fraction',.8))
    if not 0<threshold<=1:raise InputError('max_confounding_fraction must be in (0,1]')
    tau_scale=float(payload.get('tau_prior_scale',np.sqrt(np.trace(R)/n)))
    if not math.isfinite(tau_scale) or tau_scale<=0:raise InputError('tau_prior_scale must be positive')
    pts=int(payload.get('grid_points',240));
    if pts<80 or pts>1200:raise InputError('grid_points must be in [80,1200]')
    base_scale=math.sqrt(max(np.trace(R)/n,np.finfo(float).tiny));tau_min=float(payload.get('tau_min',max(base_scale*1e-5,np.finfo(float).tiny)));tau_max=float(payload.get('tau_max',max(8*tau_scale,4*base_scale)))
    if not 0<tau_min<tau_max:raise InputError('invalid tau grid bounds')
    eta=np.linspace(math.log(tau_min),math.log(tau_max),pts);taus=np.exp(eta)
    resid0=y-(A@m0+off);Cprior=A@P0@A.T
    logw=np.empty(pts);means=[];covs=[]
    P0i=linalg.inv(P0)
    for i,tau in enumerate(taus):
        S=R+(tau*tau)*K
        try:cf=linalg.cho_factor(S,lower=True,check_finite=False)
        except Exception as exc:raise InputError('total likelihood covariance not positive definite on tau grid') from exc
        SiA=linalg.cho_solve(cf,A,check_finite=False);Sir=linalg.cho_solve(cf,y-off,check_finite=False);prec=P0i+A.T@SiA;P=linalg.inv(prec);P=(P+P.T)/2;m=P@(P0i@m0+A.T@Sir);means.append(m);covs.append(P)
        V=S+Cprior
        cfv=linalg.cho_factor(V,lower=True,check_finite=False);q=float(resid0@linalg.cho_solve(cfv,resid0,check_finite=False));ld=2*float(np.log(np.diag(cfv[0])).sum())
        loglike=-.5*(n*math.log(2*math.pi)+ld+q)
        # half-normal prior on tau plus Jacobian d tau / d eta = tau
        logprior=.5*math.log(2/math.pi)-math.log(tau_scale)-.5*(tau/tau_scale)**2
        logw[i]=loglike+logprior+eta[i]
    lw=logw-special.logsumexp(logw);w=np.exp(lw);M=np.vstack(means);theta_mean=w@M
    theta_cov=np.zeros((d,d))
    for wi,mi,Pi in zip(w,M,covs):
        dm=mi-theta_mean;theta_cov+=wi*(Pi+np.outer(dm,dm))
    tau_mean=float(w@taus);tau_var=float(w@((taus-tau_mean)**2));cdf=np.cumsum(w)
    def q(prob):return float(np.interp(prob,cdf,taus))
    boundary_mass=float(w[0]+w[-1]);status='POSTERIOR_QUALIFIED'
    issues=[]
    if frac>threshold:
        status='POSTERIOR_PROVISIONAL_IDENTIFIABILITY';issues.append({'code':'PARAMETER_DISCREPANCY_CONFOUNDING','confounding_fraction':frac,'threshold':threshold})
    if boundary_mass>.02:
        status='POSTERIOR_PROVISIONAL_GRID_BOUNDARY';issues.append({'code':'TAU_GRID_BOUNDARY_MASS','boundary_mass':boundary_mass})
    return {'engine_version':ENGINE_VERSION,'kind':'linear_gaussian_joint_parameter_discrepancy','status':status,'posterior_parameter_mean':theta_mean.tolist(),'posterior_parameter_covariance':theta_cov.tolist(),'posterior_parameter_sd':np.sqrt(np.diag(theta_cov)).tolist(),
            'tau_posterior_mean':tau_mean,'tau_posterior_sd':math.sqrt(max(0,tau_var)),'tau_equal_tailed_95':[q(.025),q(.975)],'tau_grid_bounds':[tau_min,tau_max],'tau_boundary_mass':boundary_mass,
            'parameter_discrepancy_confounding_fraction':frac,'max_confounding_fraction':threshold,'issues':issues,
            'observation_covariance':R.tolist(),'discrepancy_kernel':K.tolist(),'tau_prior':{'kind':'half_normal','scale':tau_scale},
            'claim_boundary':'Joint inference is conditional on the linear model, normal parameter prior, half-normal tau prior, observation covariance, and supplied discrepancy kernel. High parameter/discrepancy confounding remains provisional rather than being hidden.'}
