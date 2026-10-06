#!/usr/bin/env python3
"""Bayesian calibration with explicit model-discrepancy covariance for v0.7.

Provides an exact linear-Gaussian route and a general random-walk Metropolis route for
safe expression models. Observation covariance and model-discrepancy covariance remain
separate inputs; discrepancy is never inferred or absorbed silently.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from scipy import linalg

import joint_distribution_engine as joint
import posterior_diagnostics as postdiag

ENGINE_VERSION = "0.7.0"
MAX_OBSERVATIONS = 20_000
MAX_PARAMETERS = 32
MAX_CHAIN_VALUES = 3_000_000


class InputError(ValueError):
    pass


def _spd(c: Any, n: int, name: str) -> np.ndarray:
    a=np.asarray(c,dtype=float)
    if a.shape!=(n,n) or not np.isfinite(a).all():raise InputError(f'{name} must be finite {n}x{n}')
    try:a=joint.validate_covariance(a,name)
    except Exception as exc:raise InputError(str(exc)) from exc
    ev=np.linalg.eigvalsh(a);scale=float(np.max(np.abs(ev))) if ev.size else 0;tol=(1e-12+1e-10)*scale if scale>0 else 0
    if np.any(ev<=tol):raise InputError(f'{name} must be positive definite for calibration likelihood')
    return a


def _obs_cov(payload:Mapping[str,Any],n:int):
    if 'observation_covariance' in payload: R=_spd(payload['observation_covariance'],n,'observation_covariance')
    elif 'observation_sd' in payload:
        sd=np.asarray(payload['observation_sd'],dtype=float)
        if sd.ndim==0:sd=np.full(n,float(sd))
        if sd.shape!=(n,) or np.any(sd<=0) or not np.isfinite(sd).all():raise InputError('observation_sd must be positive scalar or length n')
        R=np.diag(sd*sd)
    else:raise InputError('supply observation_covariance or observation_sd')
    D=np.zeros((n,n))
    if payload.get('discrepancy_covariance') is not None:
        D=np.asarray(payload['discrepancy_covariance'],dtype=float)
        if D.shape!=(n,n):raise InputError('discrepancy_covariance dimension mismatch')
        try:D=joint.validate_covariance(D,'discrepancy_covariance')
        except Exception as exc:raise InputError(str(exc)) from exc
    S=R+D
    S=_spd(S,n,'total likelihood covariance')
    return R,D,S


def linear_gaussian(payload:Mapping[str,Any])->dict[str,Any]:
    y=np.asarray(payload.get('observed'),dtype=float);A=np.asarray(payload.get('design_matrix'),dtype=float)
    if y.ndim!=1 or y.size<1 or y.size>MAX_OBSERVATIONS or not np.isfinite(y).all():raise InputError('observed must be finite vector')
    n=y.size
    if A.ndim!=2 or A.shape[0]!=n or A.shape[1]<1 or A.shape[1]>MAX_PARAMETERS or not np.isfinite(A).all():raise InputError('design_matrix mismatch')
    d=A.shape[1];offset=np.asarray(payload.get('offset',np.zeros(n)),dtype=float)
    if offset.ndim==0:offset=np.full(n,float(offset))
    if offset.shape!=(n,) or not np.isfinite(offset).all():raise InputError('offset mismatch')
    m0=np.asarray(payload.get('prior_mean'),dtype=float);P0=np.asarray(payload.get('prior_covariance'),dtype=float)
    if m0.shape!=(d,):raise InputError('prior_mean mismatch')
    P0=_spd(P0,d,'prior_covariance');R,D,S=_obs_cov(payload,n)
    cfS=linalg.cho_factor(S,lower=True);SiA=linalg.cho_solve(cfS,A);Sir=linalg.cho_solve(cfS,y-offset)
    cfP=linalg.cho_factor(P0,lower=True);P0i=linalg.cho_solve(cfP,np.eye(d));prec=P0i+A.T@SiA
    P=linalg.inv(prec);P=(P+P.T)/2;m=P@(P0i@m0+A.T@Sir)
    pred=A@m+offset;param_pred=A@P@A.T
    return {'engine_version':ENGINE_VERSION,'kind':'linear_gaussian_bayesian_calibration','status':'POSTERIOR_EXACT',
            'posterior_mean':m.tolist(),'posterior_covariance':P.tolist(),'posterior_sd':np.sqrt(np.diag(P)).tolist(),
            'posterior_predictive_mean_at_observations':pred.tolist(),'parameter_predictive_covariance':param_pred.tolist(),
            'observation_covariance':R.tolist(),'model_discrepancy_covariance':D.tolist(),'total_likelihood_covariance':S.tolist(),
            'claim_boundary':'The posterior is conditional on the supplied linear model, prior, observation covariance, and explicit discrepancy covariance.'}


def _prior_logpdf(theta:np.ndarray,names,priors):
    lp=0.0
    for i,nm in enumerate(names):
        s=dict(priors.get(nm) or {});kind=str(s.get('kind','normal')).lower();x=float(theta[i])
        if kind in {'normal','gaussian'}:
            mu=float(s.get('mean',0));sd=float(s.get('std',1))
            if sd<=0:raise InputError(f'prior {nm}: std must be >0')
            z=(x-mu)/sd;lp+=-0.5*z*z-math.log(sd)-0.5*math.log(2*math.pi)
        elif kind=='uniform':
            lo=float(s.get('lower'));hi=float(s.get('upper'))
            if not lo<hi:raise InputError(f'prior {nm}: invalid uniform bounds')
            if not lo<=x<=hi:return -math.inf
            lp+=-math.log(hi-lo)
        elif kind in {'lognormal','log_normal'}:
            mu=float(s.get('log_mean',0));sd=float(s.get('log_std',1))
            if sd<=0 or x<=0:return -math.inf
            z=(math.log(x)-mu)/sd;lp+=-0.5*z*z-math.log(sd)-math.log(x)-0.5*math.log(2*math.pi)
        else:raise InputError(f'unsupported prior kind {kind!r}')
    return float(lp)


def _initial_from_priors(names,priors,rng):
    out=[]
    for nm in names:
        s=dict(priors.get(nm) or {});kind=str(s.get('kind','normal')).lower()
        if kind in {'normal','gaussian'}:out.append(rng.normal(float(s.get('mean',0)),float(s.get('std',1))))
        elif kind=='uniform':out.append(rng.uniform(float(s['lower']),float(s['upper'])))
        elif kind in {'lognormal','log_normal'}:out.append(math.exp(rng.normal(float(s.get('log_mean',0)),float(s.get('log_std',1)))))
        else:raise InputError(f'unsupported prior kind {kind!r}')
    return np.asarray(out,dtype=float)


def metropolis(payload:Mapping[str,Any])->dict[str,Any]:
    names=list(payload.get('parameter_names') or []);d=len(names)
    if not 1<=d<=MAX_PARAMETERS or len(set(names))!=d:raise InputError('parameter_names must be unique and bounded')
    priors=dict(payload.get('priors') or {})
    if set(names)-set(priors):raise InputError('one explicit prior required per parameter')
    y=np.asarray(payload.get('observed'),dtype=float)
    if y.ndim!=1 or y.size<2 or y.size>MAX_OBSERVATIONS or not np.isfinite(y).all():raise InputError('observed must be finite vector')
    n=y.size;R,D,S=_obs_cov(payload,n);cf=linalg.cho_factor(S,lower=True);logdet=2*float(np.sum(np.log(np.diag(cf[0]))))
    data=dict(payload.get('data') or {})
    for k,v in data.items():
        a=np.asarray(v,dtype=float)
        if a.shape!=(n,) or not np.isfinite(a).all():raise InputError(f'data[{k}] must be finite length n')
        data[k]=a
    expr=str(payload.get('model_expression','')).strip()
    if not expr:raise InputError('model_expression required')
    node=joint._compile(expr)
    def prediction(theta):
        env={**data,**{nm:float(theta[i]) for i,nm in enumerate(names)}}
        with np.errstate(all='ignore'):p=np.asarray(joint._eval(node,env),dtype=float)
        if p.ndim==0:p=np.full(n,float(p))
        if p.shape!=(n,) or not np.isfinite(p).all():raise InputError('model expression produced invalid predictions')
        return p
    def logpost(theta):
        lp=_prior_logpdf(theta,names,priors)
        if not math.isfinite(lp):return -math.inf
        try:r=y-prediction(theta)
        except Exception:return -math.inf
        q=float(r@linalg.cho_solve(cf,r,check_finite=False))
        return lp-0.5*(n*math.log(2*math.pi)+logdet+q)
    chains=int(payload.get('chains',4));draws=int(payload.get('draws',2000));warmup=int(payload.get('warmup',1000));seed=int(payload.get('seed',1729))
    if chains<2 or chains>16 or draws<200 or warmup<100 or chains*(draws+warmup)*d>MAX_CHAIN_VALUES:raise InputError('invalid/oversized MCMC settings')
    prop=np.asarray(payload.get('proposal_sd',[0.2]*d),dtype=float)
    if prop.ndim==0:prop=np.full(d,float(prop))
    if prop.shape!=(d,) or np.any(prop<=0) or not np.isfinite(prop).all():raise InputError('proposal_sd must be positive and aligned')
    rng=np.random.Generator(np.random.PCG64(seed));out=np.empty((chains,warmup+draws,d));acc=[]
    initials=payload.get('initial_chains')
    if initials is not None:
        ini=np.asarray(initials,dtype=float)
        if ini.shape!=(chains,d) or not np.isfinite(ini).all():raise InputError('initial_chains shape mismatch')
    else:ini=np.vstack([_initial_from_priors(names,priors,rng) for _ in range(chains)])
    for c in range(chains):
        cur=ini[c].copy();lp=logpost(cur)
        if not math.isfinite(lp):raise InputError('initial chain state has non-finite posterior density')
        accepted=0
        for t in range(warmup+draws):
            cand=cur+rng.normal(scale=prop,size=d);clp=logpost(cand)
            if math.isfinite(clp) and math.log(rng.random()) < min(0.0,clp-lp):cur=cand;lp=clp;accepted+=1
            out[c,t]=cur
        acc.append(accepted/(warmup+draws))
    diag_payload={'chains':out.tolist(),'warmup':warmup,'parameter_names':names,
                  'max_rhat':float(payload.get('max_rhat',1.01)),'min_bulk_ess':float(payload.get('min_bulk_ess',400)),'min_tail_ess':float(payload.get('min_tail_ess',400))}
    diagnostics=postdiag.diagnose_mcmc(diag_payload)
    post=out[:,warmup:,:].reshape(-1,d);pm=post.mean(axis=0);pc=np.atleast_2d(np.cov(post,rowvar=False,ddof=1))
    # Parameter-only predictive dispersion at observed design points, evaluated on a bounded subsample.
    take=min(2000,len(post));idx=np.linspace(0,len(post)-1,take,dtype=int);P=np.vstack([prediction(post[i]) for i in idx]);pred_mean=P.mean(axis=0);pred_cov=np.atleast_2d(np.cov(P,rowvar=False,ddof=1))
    status='POSTERIOR_QUALIFIED' if diagnostics['status']=='QUALIFIED' else 'POSTERIOR_NOT_QUALIFIED'
    result={'engine_version':ENGINE_VERSION,'kind':'random_walk_metropolis_calibration','status':status,'parameter_names':names,
            'posterior_mean':pm.tolist(),'posterior_covariance':pc.tolist(),'posterior_sd':np.sqrt(np.diag(pc)).tolist(),
            'acceptance_rates':acc,'diagnostics':diagnostics,
            'posterior_predictive_mean_parameter_only':pred_mean.tolist(),'posterior_predictive_covariance_parameter_only':pred_cov.tolist(),
            'observation_covariance':R.tolist(),'model_discrepancy_covariance':D.tolist(),
            'chain_shape':[int(chains),int(warmup+draws),int(d)],
            'claim_boundary':'Posterior qualification assesses the generated chains, not scientific adequacy of the calibration model. Discrepancy covariance is explicit and fixed; it is not silently estimated from residuals.'}
    if bool(payload.get('return_chains',False)):
        result['chains_draws']=out.tolist()
    return result
