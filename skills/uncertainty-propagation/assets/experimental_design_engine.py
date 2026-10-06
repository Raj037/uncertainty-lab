#!/usr/bin/env python3
"""Uncertainty-aware experimental design / value of information for v0.8.

Qualified route: finite candidate measurements under a linear-Gaussian parameter model.
Candidate posterior covariance is exact and independent of the as-yet-unobserved result.
Supports D-like information gain, A-like trace reduction, c-optimal target variance
reduction, and expected binary decision-entropy reduction for a linear target.
"""
from __future__ import annotations
import math
from typing import Any,Mapping
import numpy as np
from scipy import linalg, stats
from numpy.polynomial.hermite import hermgauss
import joint_distribution_engine as joint

ENGINE_VERSION='0.8.0'
MAX_DIM=128
MAX_CANDIDATES=5000
class InputError(ValueError):pass

def _state(p):
    m=np.asarray(p.get('parameter_mean'),dtype=float);P=np.asarray(p.get('parameter_covariance'),dtype=float)
    if m.ndim!=1 or not 1<=len(m)<=MAX_DIM or P.shape!=(len(m),len(m)) or not np.isfinite(m).all():raise InputError('parameter_mean/covariance mismatch')
    try:P=joint.validate_covariance(P,'parameter_covariance')
    except Exception as exc:raise InputError(str(exc)) from exc
    diag=np.diag(P)
    if np.any(diag<=0):raise InputError('experimental-design covariance must have positive variances')
    sd=np.sqrt(diag);R=P/np.outer(sd,sd);R=(R+R.T)/2;ev=np.linalg.eigvalsh(R);tol=1e-12+1e-10*max(1.0,float(np.max(np.abs(ev))))
    if ev[0] <= tol:raise InputError('experimental-design covariance must be positive definite on a variance-normalized basis')
    return m,P

def _candidates(p,d):
    rows=list(p.get('candidates') or [])
    if not 1<=len(rows)<=MAX_CANDIDATES:raise InputError(f'candidates must contain 1..{MAX_CANDIDATES} entries')
    out=[];names=set()
    for i,r0 in enumerate(rows):
        r=dict(r0);name=str(r.get('name',f'candidate_{i}'))
        if name in names:raise InputError('candidate names must be unique')
        names.add(name);h=np.asarray(r.get('sensitivity'),dtype=float);sd=float(r.get('noise_std',0));cost=float(r.get('cost',1.0))
        if h.shape!=(d,) or not np.isfinite(h).all() or not math.isfinite(sd) or sd<=0 or not math.isfinite(cost) or cost<=0:raise InputError(f'invalid candidate {name}')
        out.append({'name':name,'h':h,'noise_var':sd*sd,'noise_std':sd,'cost':cost,'metadata':r.get('metadata')})
    return out

def _update(P,h,r):
    ph=P@h;v=float(h@ph+r)
    if not math.isfinite(v) or v<=0:raise InputError('nonpositive predictive measurement variance')
    Q=P-np.outer(ph,ph)/v;Q=(Q+Q.T)/2
    return Q,v,ph

def _logdet(P):
    sign,ld=np.linalg.slogdet(P)
    if sign<=0:raise InputError('covariance lost positive definiteness')
    return float(ld)

def _Hbin(p):
    p=min(1-1e-15,max(1e-15,float(p)));return -p*math.log(p)-(1-p)*math.log(1-p)

def _decision_entropy(m,P,h,r,target,threshold):
    c=np.asarray(target,dtype=float);mt=float(c@m);vt=float(c@P@c)
    if vt<=0:raise InputError('target variance must be positive')
    ph=P@h;vy=float(h@ph+r);cov=float(c@ph);vpost=vt-cov*cov/vy;vpost=max(vpost,np.finfo(float).tiny)
    pprior=float(stats.norm.cdf((threshold-mt)/math.sqrt(vt)));H0=_Hbin(pprior)
    # Gauss-Hermite integrates over the predictive standard normal measurement innovation.
    x,w=hermgauss(32);Hs=[]
    gain=cov/math.sqrt(vy)
    for xx in x:
        mpost=mt+gain*math.sqrt(2)*float(xx);pp=float(stats.norm.cdf((threshold-mpost)/math.sqrt(vpost)));Hs.append(_Hbin(pp))
    EH=float(np.dot(w,np.asarray(Hs))/math.sqrt(math.pi));return H0-EH,pprior,vpost

def score_candidates(payload:Mapping[str,Any], *, covariance=None, mean=None, exclude=None):
    m,P=_state(payload) if covariance is None else (np.asarray(mean,dtype=float),np.asarray(covariance,dtype=float));rows=_candidates(payload,len(m));exclude=set(exclude or [])
    target=payload.get('target_sensitivity');c=None if target is None else np.asarray(target,dtype=float)
    if c is not None and (c.shape!=(len(m),) or not np.isfinite(c).all()):raise InputError('target_sensitivity mismatch')
    threshold=payload.get('decision_threshold');objective=str(payload.get('objective','information_gain')).lower();per_cost=bool(payload.get('per_cost',False))
    if objective in {'target_variance_reduction','decision_entropy_reduction'} and c is None:raise InputError(f'{objective} requires target_sensitivity')
    if objective=='decision_entropy_reduction' and threshold is None:raise InputError('decision_entropy_reduction requires decision_threshold')
    base_ld=_logdet(P);base_trace=float(np.trace(P));base_tvar=None if c is None else float(c@P@c);scores=[]
    for r in rows:
        if r['name'] in exclude:continue
        Q,vy,ph=_update(P,r['h'],r['noise_var']);ig=.5*(base_ld-_logdet(Q));tr=base_trace-float(np.trace(Q));tvr=None if c is None else base_tvar-float(c@Q@c);de=None;priorp=None
        if c is not None and threshold is not None:de,priorp,_=_decision_entropy(m,P,r['h'],r['noise_var'],c,float(threshold))
        metric={'information_gain':ig,'trace_reduction':tr,'target_variance_reduction':tvr,'decision_entropy_reduction':de}.get(objective)
        if metric is None:raise InputError('objective must be information_gain, trace_reduction, target_variance_reduction, or decision_entropy_reduction')
        utility=float(metric)/(r['cost'] if per_cost else 1.0)
        scores.append({'name':r['name'],'score':utility,'raw_objective_value':float(metric),'cost':r['cost'],'information_gain_nats':ig,'trace_reduction':tr,'target_variance_reduction':tvr,'decision_entropy_reduction_nats':de,'prior_decision_probability_below_threshold':priorp,'predictive_measurement_variance':vy,'posterior_covariance':Q.tolist(),'metadata':r['metadata']})
    scores.sort(key=lambda x:x['score'],reverse=True)
    return {'engine_version':ENGINE_VERSION,'kind':'linear_gaussian_candidate_design','objective':objective,'per_cost':per_cost,'ranked_candidates':scores,'best_candidate':scores[0] if scores else None,
            'claim_boundary':'Scores are exact only under the declared linear-Gaussian parameter/measurement model. Cost normalization is utility-per-cost, not a proof of optimal project-level value.'}

def greedy_batch(payload:Mapping[str,Any])->dict[str,Any]:
    m,P=_state(payload);k=int(payload.get('batch_size',1));rows=_candidates(payload,len(m))
    if not 1<=k<=min(50,len(rows)):raise InputError('batch_size out of range')
    chosen=[];steps=[];Q=P.copy()
    for step in range(k):
        r=score_candidates(payload,covariance=Q,mean=m,exclude=chosen);best=r['best_candidate']
        if best is None:break
        chosen.append(best['name']);Q=np.asarray(best['posterior_covariance'],dtype=float);steps.append({'step':step+1,'selected':best['name'],'score':best['score'],'raw_objective_value':best['raw_objective_value'],'posterior_trace':float(np.trace(Q)),'posterior_logdet':_logdet(Q)})
    return {'engine_version':ENGINE_VERSION,'kind':'greedy_linear_gaussian_batch_design','selected_candidates':chosen,'steps':steps,'prior_covariance':P.tolist(),'posterior_covariance_after_batch':Q.tolist(),'total_trace_reduction':float(np.trace(P)-np.trace(Q)),'total_information_gain_nats':.5*(_logdet(P)-_logdet(Q)),
            'claim_boundary':'Greedy batch selection is sequentially optimal only at each local step; it is not a proof of globally optimal combinatorial design.'}
