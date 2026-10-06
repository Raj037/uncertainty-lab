#!/usr/bin/env python3
"""Active-learning GP reliability for expensive limit-state models (v0.8).

Uses an AK-MCS-style U acquisition |mu_g|/sigma_g on a fixed candidate population.
The production route returns the next candidate points to evaluate. A separate expression-
oracle route exists for deterministic regression tests and inexpensive analytic models.
"""
from __future__ import annotations
import math
from typing import Any,Mapping
import numpy as np
import surrogate_engine as sg
import joint_distribution_engine as joint

ENGINE_VERSION='0.8.0'
MAX_POOL=300_000
class InputError(ValueError):pass

def _xy(p):
    X=np.asarray(p.get('X'),dtype=float);g=np.asarray(p.get('g'),dtype=float)
    if X.ndim!=2 or g.ndim!=1 or len(g)!=len(X) or len(g)<6 or not np.isfinite(X).all() or not np.isfinite(g).all():raise InputError('X/g must be >=6 aligned finite evaluated limit-state points')
    return X,g

def _pool(p,d):
    if p.get('candidate_X') is not None:
        C=np.asarray(p['candidate_X'],dtype=float)
        if C.ndim!=2 or C.shape[1]!=d or len(C)<100 or len(C)>MAX_POOL or not np.isfinite(C).all():raise InputError('candidate_X mismatch/size')
        return C,{'kind':'provided_candidate_pool'}
    dist=dict(p.get('input_distribution') or {})
    if not dist:raise InputError('supply candidate_X or input_distribution')
    dist['n']=int(p.get('candidate_n',20000));dist['seed']=int(p.get('candidate_seed',1729))
    names,C,meta=joint.generate_inputs(dist)
    if C.shape[1]!=d or len(C)>MAX_POOL:raise InputError('input_distribution dimension/pool mismatch')
    return C,meta

def _fit_predict(X,g,C,p):
    pp={'noise_std':float(p.get('gp_noise_std',1e-8))}
    m=sg._gp_fit_raw(X,g,pp);mu,var=sg._gp_predict(m,C);sd=np.sqrt(np.maximum(0,var));return m,mu,sd

def _summary(mu,sd,p):
    z=float(p.get('classification_z',2.0))
    if z<=0 or not math.isfinite(z):raise InputError('classification_z must be positive')
    pf=float(np.mean(mu<=0));lower=float(np.mean(mu+z*sd<=0));upper=float(np.mean(mu-z*sd<=0));amb=float(np.mean((mu-z*sd<=0)&(mu+z*sd>0)))
    return {'failure_probability_surrogate_mean_classifier':pf,'classification_probability_lower':lower,'classification_probability_upper':upper,'ambiguous_candidate_fraction':amb,'classification_z':z,'interval_width':upper-lower}

def select_points(payload:Mapping[str,Any])->dict[str,Any]:
    X,g=_xy(payload);C,meta=_pool(payload,X.shape[1]);_,mu,sd=_fit_predict(X,g,C,payload);U=np.divide(np.abs(mu),sd,out=np.full_like(sd,np.inf),where=sd>0)
    k=int(payload.get('batch_size',1));
    if k<1 or k>50:raise InputError('batch_size must be in [1,50]')
    idx=np.argsort(U)[:k];summ=_summary(mu,sd,payload);threshold=float(payload.get('u_stop_threshold',2.0));width=float(payload.get('max_classification_interval_width',0.002))
    converged=bool(float(np.min(U))>=threshold and summ['interval_width']<=width)
    return {'engine_version':ENGINE_VERSION,'kind':'gp_active_learning_reliability_selection','status':'RELIABILITY_CONVERGED' if converged else 'NEEDS_EVALUATIONS',
            'next_candidate_indices':idx.tolist(),'next_candidate_points':C[idx].tolist(),'next_candidate_U':U[idx].tolist(),'minimum_U':float(np.min(U)),'u_stop_threshold':threshold,
            'evaluated_count':len(X),'candidate_count':len(C),'candidate_source':meta,**summ,
            'claim_boundary':'The returned reliability interval reflects GP classification uncertainty over the fixed candidate population, not all possible surrogate/model-form uncertainty. New simulator values must be supplied before production iteration continues.'}

def active_learn_expression(payload:Mapping[str,Any])->dict[str,Any]:
    X,g=_xy(payload);C,meta=_pool(payload,X.shape[1]);expr=str(payload.get('oracle_expression','')).strip();names=list(payload.get('variables') or [f'x{i}' for i in range(X.shape[1])])
    if not expr or len(names)!=X.shape[1]:raise InputError('oracle_expression and aligned variables required')
    node=joint._compile(expr);budget=int(payload.get('evaluation_budget',20));
    if budget<0 or budget>500:raise InputError('evaluation_budget must be in [0,500]')
    history=[]
    def oracle(P):
        env={n:P[:,i] for i,n in enumerate(names)}
        with np.errstate(all='ignore'):v=np.asarray(joint._eval(node,env),dtype=float)
        if v.ndim==0:v=np.full(len(P),float(v))
        if v.shape!=(len(P),) or not np.isfinite(v).all():raise InputError('oracle expression invalid')
        return v
    for step in range(budget+1):
        _,mu,sd=_fit_predict(X,g,C,payload);U=np.divide(np.abs(mu),sd,out=np.full_like(sd,np.inf),where=sd>0);summ=_summary(mu,sd,payload);threshold=float(payload.get('u_stop_threshold',2.0));width=float(payload.get('max_classification_interval_width',0.002));conv=bool(float(np.min(U))>=threshold and summ['interval_width']<=width)
        history.append({'step':step,'evaluated_count':len(X),'minimum_U':float(np.min(U)),'interval_width':summ['interval_width'],'pf':summ['failure_probability_surrogate_mean_classifier']})
        if conv or step==budget:break
        i=int(np.argmin(U));x=C[i:i+1];y=oracle(x);X=np.vstack([X,x]);g=np.r_[g,y]
        # Remove selected candidate to prevent duplicate acquisition.
        C=np.delete(C,i,axis=0)
    exact=oracle(C);direct=float(np.mean(exact<=0)) if len(C) else None
    return {'engine_version':ENGINE_VERSION,'kind':'gp_active_learning_reliability_expression_oracle','status':'RELIABILITY_CONVERGED' if conv else 'BUDGET_EXHAUSTED','evaluated_count':len(X),'new_evaluations':len(X)-len(np.asarray(payload['X'])),
            'candidate_remaining':len(C),'history':history,'direct_oracle_failure_fraction_on_remaining_pool':direct,**summ,
            'claim_boundary':'Expression-oracle mode is a validation/convenience route. For an expensive external simulator use select_points, evaluate those points externally, append X/g, and rerun.'}
