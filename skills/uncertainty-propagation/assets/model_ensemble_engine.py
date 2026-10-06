#!/usr/bin/env python3
"""Structural model uncertainty / model ensembles for Uncertainty Lab v1.0.

Combines explicit model-level predictive means/covariances using either supplied
weights or prior-adjusted log evidences. Reports within-model and between-model
covariance separately. Optional interval model weights produce robust scalar
bounds via linear programming rather than silently selecting one weighting.
"""
from __future__ import annotations
import math
from typing import Any, Mapping
import numpy as np
from scipy.optimize import linprog
import joint_distribution_engine as joint
ENGINE_VERSION='1.0.0'
MAX_MODELS=64
MAX_OUTPUTS=64
class InputError(ValueError): pass


def _models(payload):
    rows=list(payload.get('models') or [])
    if not 1<=len(rows)<=MAX_MODELS: raise InputError(f'models must contain 1..{MAX_MODELS} entries')
    names=[]; means=[]; covs=[]; loge=[]; explicit=[]; event=[]
    d=None
    for i,r0 in enumerate(rows):
        r=dict(r0); name=str(r.get('name',f'model_{i}'))
        if name in names: raise InputError('model names must be unique')
        mu=np.asarray(r.get('mean'),dtype=float)
        if mu.ndim!=1 or not 1<=len(mu)<=MAX_OUTPUTS or not np.isfinite(mu).all(): raise InputError('model mean invalid')
        if d is None:d=len(mu)
        if len(mu)!=d: raise InputError('model output dimensions mismatch')
        C=joint.validate_covariance(r.get('covariance'),f'{name} covariance')
        if C.shape!=(d,d): raise InputError('model covariance dimension mismatch')
        names.append(name);means.append(mu);covs.append(C);loge.append(r.get('log_evidence'));explicit.append(r.get('weight'));event.append(r.get('event_probability'))
    return names,np.vstack(means),covs,loge,explicit,event


def _weights(payload,loge,explicit):
    m=len(loge)
    if all(x is not None for x in explicit):
        w=np.asarray(explicit,dtype=float)
        if np.any(w<0) or not np.isfinite(w).all() or w.sum()<=0: raise InputError('explicit model weights invalid')
        return w/w.sum(),'explicit_weights'
    if all(x is not None for x in loge):
        le=np.asarray(loge,dtype=float); prior=np.asarray(payload.get('prior_model_weights',[1/m]*m),dtype=float)
        if prior.shape!=(m,) or np.any(prior<=0) or not np.isfinite(prior).all(): raise InputError('prior_model_weights invalid')
        z=le+np.log(prior);z-=np.max(z);w=np.exp(z);return w/w.sum(),'prior_adjusted_log_evidence'
    raise InputError('supply either weight for every model or log_evidence for every model')


def combine(payload:Mapping[str,Any])->dict[str,Any]:
    names,M,Cs,loge,explicit,event=_models(payload);w,basis=_weights(payload,loge,explicit);mu=w@M
    within=sum((w[i]*Cs[i] for i in range(len(w))),np.zeros_like(Cs[0]));between=np.zeros_like(within)
    for i in range(len(w)):
        d=(M[i]-mu)[:,None];between+=w[i]*(d@d.T)
    total=within+between
    out={'engine_version':ENGINE_VERSION,'kind':'model_ensemble_mixture_moments','status':'PASS','model_names':names,'weights':w.tolist(),'weight_basis':basis,
         'mixture_mean':mu.tolist(),'within_model_covariance':within.tolist(),'between_model_covariance':between.tolist(),'total_covariance':total.tolist(),
         'within_trace':float(np.trace(within)),'between_trace':float(np.trace(between)),'total_trace':float(np.trace(total)),
         'claim_boundary':'Weights are conditional on the supplied model set and evidence/weights. Bayesian model averaging does not prove that the model list is complete or mutually exclusive.'}
    if all(x is not None for x in event): out['mixture_event_probability']=float(w@np.asarray(event,dtype=float))
    return out


def _weight_polytope(payload,m):
    ints=np.asarray(payload.get('weight_intervals'),dtype=float)
    if ints.shape!=(m,2) or not np.isfinite(ints).all() or np.any(ints<0) or np.any(ints[:,0]>ints[:,1]): raise InputError('weight_intervals must be Mx2 lower/upper bounds')
    if ints[:,0].sum()>1+1e-12 or ints[:,1].sum()<1-1e-12: raise InputError('weight intervals cannot satisfy sum(weights)=1')
    return ints

def _lp_bounds(values,ints):
    v=np.asarray(values,dtype=float);bounds=[tuple(x) for x in ints];Aeq=np.ones((1,len(v)));beq=[1.0]
    lo=linprog(v,A_eq=Aeq,b_eq=beq,bounds=bounds,method='highs');hi=linprog(-v,A_eq=Aeq,b_eq=beq,bounds=bounds,method='highs')
    if not lo.success or not hi.success: raise InputError('model-weight interval optimization failed')
    return float(lo.fun),float(-hi.fun),lo.x.tolist(),hi.x.tolist()

def robust_scalar_bounds(payload:Mapping[str,Any])->dict[str,Any]:
    names,M,Cs,loge,explicit,event=_models(payload);ints=_weight_polytope(payload,len(names));kind=str(payload.get('quantity','output_mean')).lower();idx=int(payload.get('output_index',0))
    if kind=='output_mean':
        if not 0<=idx<M.shape[1]:raise InputError('output_index out of range')
        vals=M[:,idx]
    elif kind=='event_probability':
        if any(x is None for x in event):raise InputError('every model requires event_probability')
        vals=np.asarray(event,dtype=float)
    else:raise InputError('quantity must be output_mean or event_probability')
    lo,hi,wlo,whi=_lp_bounds(vals,ints)
    return {'engine_version':ENGINE_VERSION,'kind':'robust_model_weight_bounds','status':'PASS','quantity':kind,'model_names':names,'lower':lo,'upper':hi,'lower_extremizer_weights':wlo,'upper_extremizer_weights':whi,
            'claim_boundary':'Bounds are exact over the declared model-weight polytope and supplied model-specific scalar values; they do not cover omitted models.'}
