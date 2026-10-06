#!/usr/bin/env python3
"""Multi-fidelity uncertainty quantification for Uncertainty Lab v0.9.

Two qualified routes:
1. Independent-sample control-variate estimation using paired high/low-fidelity
   pilot runs plus an independent large low-fidelity sample.
2. Autoregressive multi-fidelity GP: y_H(x)=rho*m_L(x)+delta(x), with separate
   low-fidelity and discrepancy GPs and repeated held-out HIGH-fidelity validation.

The GP route reports the independence approximation used when combining component
predictive variances; validation of high-fidelity predictions remains authoritative.
"""
from __future__ import annotations
import math
from typing import Any, Mapping
import numpy as np
import surrogate_engine as sur
import joint_distribution_engine as joint
ENGINE_VERSION='0.9.0'
class InputError(ValueError): pass


def control_variate(payload:Mapping[str,Any])->dict[str,Any]:
    H=np.asarray(payload.get('high_values'),dtype=float); Lp=np.asarray(payload.get('low_paired_values'),dtype=float); Le=np.asarray(payload.get('low_extra_values'),dtype=float)
    if H.ndim!=1 or Lp.shape!=H.shape or H.size<10 or Le.ndim!=1 or Le.size<20 or not np.isfinite(H).all() or not np.isfinite(Lp).all() or not np.isfinite(Le).all(): raise InputError('high/paired-low/independent-extra-low samples invalid')
    vL=float(np.var(Lp,ddof=1))
    if vL<=0: raise InputError('paired low-fidelity variance must be positive')
    cov=float(np.cov(H,Lp,ddof=1)[0,1]); beta=cov/vL
    Z=H-beta*Lp; est=float(Z.mean()+beta*Le.mean()); se=math.sqrt(float(np.var(Z,ddof=1))/len(Z)+beta*beta*float(np.var(Le,ddof=1))/len(Le))
    naive_se=math.sqrt(float(np.var(H,ddof=1))/len(H)); corr=float(np.corrcoef(H,Lp)[0,1])
    return {'engine_version':ENGINE_VERSION,'kind':'multifidelity_control_variate','status':'PASS','estimate':est,'standard_error_estimate':se,'naive_high_only_standard_error':naive_se,
            'estimated_beta':beta,'paired_correlation':corr,'high_runs':int(H.size),'extra_low_runs':int(Le.size),'estimated_se_reduction_factor':naive_se/se if se>0 else math.inf,
            'claim_boundary':'SE formula requires the extra low-fidelity sample to be independent of the paired high/low pilot sample; beta is estimated from the pilot and its estimation uncertainty is not separately expanded.'}


def _fit_mf(Xl,yl,Xh,yh,payload):
    low=sur._gp_fit_raw(Xl,yl,payload); ml,_=sur._gp_predict(low,Xh); den=float(ml@ml)
    if den<=0: raise InputError('low-fidelity prediction has zero norm at high-fidelity sites')
    rho=float((ml@yh)/den); resid=yh-rho*ml
    # delta GP may be close to constant; add an explicit small affine jitter only when needed.
    if float(np.std(resid,ddof=1))<=1e-14:
        return {'low':low,'rho':rho,'delta_constant':float(resid.mean()),'delta':None}
    delta=sur._gp_fit_raw(Xh,resid,payload)
    return {'low':low,'rho':rho,'delta':delta,'delta_constant':None}


def _predict_mf(model,X):
    ml,vl=sur._gp_predict(model['low'],X)
    if model['delta'] is None:
        md=np.full(len(X),model['delta_constant']); vd=np.zeros(len(X))
    else: md,vd=sur._gp_predict(model['delta'],X)
    mean=model['rho']*ml+md; var=model['rho']**2*vl+vd
    return mean,var


def autoregressive_gp(payload:Mapping[str,Any])->dict[str,Any]:
    Xl=np.asarray(payload.get('X_low'),dtype=float); yl=np.asarray(payload.get('y_low'),dtype=float); Xh=np.asarray(payload.get('X_high'),dtype=float); yh=np.asarray(payload.get('y_high'),dtype=float)
    if Xl.ndim!=2 or Xh.ndim!=2 or Xl.shape[1]!=Xh.shape[1] or yl.shape!=(len(Xl),) or yh.shape!=(len(Xh),) or len(Xl)<12 or len(Xh)<10 or not all(np.isfinite(z).all() for z in [Xl,yl,Xh,yh]): raise InputError('invalid multi-fidelity training arrays')
    runs=[]
    for seed in sur._validation_seeds(payload):
        pp=dict(payload);pp['validation_seed']=seed;tr,va=sur._split(len(yh),pp); m=_fit_mf(Xl,yl,Xh[tr],yh[tr],pp); pred,pv=_predict_mf(m,Xh[va]); runs.append({'seed':seed,'metrics':sur._metrics(yh[va],pred),'mean_predictive_variance':float(np.mean(pv)),'rho':m['rho']})
    metrics=sur._aggregate_metrics([r['metrics'] for r in runs]); status,q=sur._qualify(metrics,payload); final=_fit_mf(Xl,yl,Xh,yh,payload)
    out={'engine_version':ENGINE_VERSION,'kind':'autoregressive_multifidelity_gp','status':status,'validation_metrics':metrics,'validation_runs':runs,'qualification':q,'rho':final['rho'],
         'claim_boundary':'Qualification is against held-out high-fidelity observations. Combined predictive variance uses an independence approximation between low-GP and discrepancy-GP posterior errors.'}
    if payload.get('input_distribution') is not None:
        if status!='SURROGATE_QUALIFIED': out['propagation_status']='BLOCKED_UNQUALIFIED_SURROGATE'; return out
        d=dict(payload['input_distribution']); d['n']=int(payload.get('n',50000));d['seed']=int(payload.get('seed',1729)); names,S,_=joint.generate_inputs(d); mean,var=_predict_mf(final,S)
        out.update({'propagation_status':'PASS','input_variable_order':names,'n_propagation':len(mean),'output_mean':float(mean.mean()),'variance_due_to_input_through_multifidelity_mean':float(np.var(mean,ddof=1)),
                    'mean_surrogate_predictive_variance':float(np.mean(var)),'total_predictive_variance_approx':float(np.var(mean,ddof=1)+np.mean(var))})
    return out
