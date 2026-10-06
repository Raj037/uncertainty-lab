#!/usr/bin/env python3
"""Validated surrogate models for expensive black-box simulations (Uncertainty Lab v0.7).

Implements scalar Gaussian-process regression and polynomial-chaos expansion (PCE).
A surrogate is never promoted solely because it fits its training data: qualification
requires held-out validation against user-declared thresholds. Surrogate predictive/
validation uncertainty is reported separately from physical/input uncertainty.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import linalg, optimize
from scipy.special import eval_hermitenorm, eval_legendre, factorial

import joint_distribution_engine as joint

ENGINE_VERSION = "0.7.0"
MAX_GP_TRAIN = 3000
MAX_PCE_TERMS = 5000
MAX_DIM = 32


class InputError(ValueError):
    pass


def _xy(payload: Mapping[str, Any]):
    X = np.asarray(payload.get("X"), dtype=float)
    y = np.asarray(payload.get("y"), dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size or X.shape[0] < 8 or X.shape[1] < 1 or X.shape[1] > MAX_DIM:
        raise InputError("X/y must be aligned finite scalar-output training data with >=8 rows")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise InputError("X/y contain non-finite values")
    return X, y


def _split(n: int, payload: Mapping[str, Any]):
    frac = float(payload.get("validation_fraction", 0.2)); seed = int(payload.get("validation_seed", 1729))
    if not 0.1 <= frac <= 0.5:
        raise InputError("validation_fraction must lie in [0.1,0.5]")
    rng = np.random.Generator(np.random.PCG64(seed)); idx = rng.permutation(n)
    nv = max(2, int(round(frac*n))); nv = min(n-4, nv)
    return idx[nv:], idx[:nv]


def _validation_seeds(payload: Mapping[str, Any]) -> list[int]:
    reps=int(payload.get('validation_repeats',3));base=int(payload.get('validation_seed',1729))
    if reps<1 or reps>10: raise InputError('validation_repeats must lie in [1,10]')
    return [base+i*7919 for i in range(reps)]

def _aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    # Conservative aggregation: qualification must survive the worst held-out split.
    return {
        'rmse': float(max(float(r['rmse']) for r in rows)),
        'mae': float(max(float(r['mae']) for r in rows)),
        'nrmse_by_range': None if any(r['nrmse_by_range'] is None for r in rows) else float(max(float(r['nrmse_by_range']) for r in rows)),
        'r2': None if any(r['r2'] is None for r in rows) else float(min(float(r['r2']) for r in rows)),
        'max_abs_error': float(max(float(r['max_abs_error']) for r in rows)),
    }

def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    err = pred-y; rmse=float(np.sqrt(np.mean(err*err))); mae=float(np.mean(np.abs(err)))
    span=float(np.max(y)-np.min(y)); nrmse=None if span==0 else rmse/span
    sst=float(np.sum((y-y.mean())**2)); r2=None if sst==0 else 1-float(np.sum(err*err))/sst
    return {"rmse":rmse,"mae":mae,"nrmse_by_range":nrmse,"r2":r2,"max_abs_error":float(np.max(np.abs(err)))}


def _qualify(metrics: Mapping[str, Any], payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    gates = dict(payload.get("qualification") or {})
    if not gates:
        return "PROVISIONAL", {"reason":"No held-out qualification thresholds supplied."}
    checks={}
    if "max_nrmse" in gates:
        v=metrics.get("nrmse_by_range"); checks["max_nrmse"] = bool(v is not None and v <= float(gates["max_nrmse"]))
    if "min_r2" in gates:
        v=metrics.get("r2"); checks["min_r2"] = bool(v is not None and v >= float(gates["min_r2"]))
    if "max_abs_error" in gates:
        checks["max_abs_error"] = bool(float(metrics["max_abs_error"]) <= float(gates["max_abs_error"]))
    if not checks:
        raise InputError("qualification must contain max_nrmse, min_r2, and/or max_abs_error")
    return ("SURROGATE_QUALIFIED" if all(checks.values()) else "SURROGATE_REJECTED"), {"thresholds":gates,"checks":checks}


def _standardize_X(X: np.ndarray):
    center=X.mean(axis=0); scale=X.std(axis=0,ddof=1)
    if np.any(scale <= 0): raise InputError("GP requires non-constant training inputs")
    return (X-center)/scale,center,scale


def _sqdist(A:np.ndarray,B:np.ndarray)->np.ndarray:
    aa=np.sum(A*A,axis=1)[:,None];bb=np.sum(B*B,axis=1)[None,:]
    return np.maximum(0.0,aa+bb-2*A@B.T)


def _gp_fit_raw(X:np.ndarray,y:np.ndarray, payload:Mapping[str,Any]):
    if X.shape[0]>MAX_GP_TRAIN: raise InputError(f"GP training rows exceed {MAX_GP_TRAIN}")
    Z,xc,xs=_standardize_X(X);ym=float(y.mean());ys=float(y.std(ddof=1))
    if ys==0: raise InputError("GP output is constant; use the constant model directly")
    t=(y-ym)/ys;n=len(y)
    noise_fixed=payload.get("noise_std")
    if noise_fixed is not None:
        nf=float(noise_fixed)/ys
        if not math.isfinite(nf) or nf<0: raise InputError("noise_std must be finite nonnegative")
    # log(length), log(signal sd), [log(noise sd)]
    x0=[0.0,0.0] + ([] if noise_fixed is not None else [-5.0])
    bounds=[(-5,5),(-5,5)] + ([] if noise_fixed is not None else [(-12,-0.5)])
    D=_sqdist(Z,Z)
    def nll(th):
        ell=math.exp(float(th[0]));sf=math.exp(float(th[1]));sn=nf if noise_fixed is not None else math.exp(float(th[2]))
        K=sf*sf*np.exp(-0.5*D/(ell*ell))+(sn*sn+1e-12)*np.eye(n)
        try:
            cf=linalg.cho_factor(K,lower=True,check_finite=False);alpha=linalg.cho_solve(cf,t,check_finite=False)
        except linalg.LinAlgError:return 1e100
        return float(0.5*t@alpha+np.sum(np.log(np.diag(cf[0])))+0.5*n*math.log(2*math.pi))
    starts=[np.array(x0,dtype=float), np.array([1.0,0.0]+([] if noise_fixed is not None else [-4.0]),dtype=float), np.array([-1.0,0.0]+([] if noise_fixed is not None else [-6.0]),dtype=float)]
    candidates=[]
    for start in starts:
        rr=optimize.minimize(nll,start,method='L-BFGS-B',bounds=bounds,options={'maxiter':300,'ftol':1e-12})
        if math.isfinite(float(rr.fun)): candidates.append(rr)
    if not candidates or not any(rr.success for rr in candidates):
        # Derivative-free fallback is intentionally a different optimizer; held-out
        # validation still decides whether the resulting surrogate is admissible.
        rr=optimize.minimize(nll,np.array(x0,dtype=float),method='Powell',bounds=bounds,options={'maxiter':600,'xtol':1e-5,'ftol':1e-10})
        if math.isfinite(float(rr.fun)): candidates.append(rr)
    if not candidates: raise InputError('GP hyperparameter optimization produced no finite candidate')
    res=min(candidates,key=lambda z:float(z.fun))
    th=res.x;ell=math.exp(float(th[0]));sf=math.exp(float(th[1]));sn=nf if noise_fixed is not None else math.exp(float(th[2]))
    K=sf*sf*np.exp(-0.5*D/(ell*ell))+(sn*sn+1e-12)*np.eye(n)
    cf=linalg.cho_factor(K,lower=True,check_finite=False);alpha=linalg.cho_solve(cf,t,check_finite=False)
    return {'X':X,'Z':Z,'t':t,'xc':xc,'xs':xs,'ym':ym,'ys':ys,'ell':ell,'sf':sf,'sn':sn,'cf':cf,'alpha':alpha,'nll':float(res.fun)}


def _gp_predict(model:Mapping[str,Any],Xnew:np.ndarray):
    Xnew=np.asarray(Xnew,dtype=float)
    if Xnew.ndim!=2 or Xnew.shape[1]!=model['X'].shape[1] or not np.isfinite(Xnew).all():raise InputError('GP prediction input mismatch')
    Zn=(Xnew-model['xc'])/model['xs'];D=_sqdist(Zn,model['Z']);Kx=model['sf']**2*np.exp(-0.5*D/(model['ell']**2))
    mt=Kx@model['alpha'];v=linalg.cho_solve(model['cf'],Kx.T,check_finite=False)
    var_t=np.maximum(0.0,model['sf']**2-np.sum(Kx*v.T,axis=1))
    return model['ym']+model['ys']*mt, model['ys']**2*var_t


def gaussian_process(payload:Mapping[str,Any])->dict[str,Any]:
    X,y=_xy(payload);runs=[]
    for seed in _validation_seeds(payload):
        pp=dict(payload);pp['validation_seed']=seed;tr,va=_split(len(y),pp);m=_gp_fit_raw(X[tr],y[tr],pp);pred,pvar=_gp_predict(m,X[va]);met=_metrics(y[va],pred)
        runs.append({'seed':seed,'n_train':int(len(tr)),'n_validation':int(len(va)),'metrics':met,'mean_predictive_variance':float(np.mean(pvar)),
                     'hyperparameters_standardized':{'length_scale':m['ell'],'signal_sd':m['sf'],'noise_sd':m['sn']}})
    metrics=_aggregate_metrics([x['metrics'] for x in runs]);status,q=_qualify(metrics,payload)
    return {'engine_version':ENGINE_VERSION,'kind':'gaussian_process_rbf','status':status,'validation_repeats':len(runs),
            'validation_metrics':metrics,'validation_runs':runs,'qualification':q,
            'claim_boundary':'Qualification uses the worst metric across deterministic held-out splits. GP predictive variance is surrogate/model uncertainty conditional on GP assumptions, not physical measurement uncertainty.'}


def _multiindices(d:int,p:int):
    out=[]
    for total in range(p+1):
        for comb in itertools.product(range(total+1),repeat=d):
            if sum(comb)==total:out.append(comb)
            if len(out)>MAX_PCE_TERMS:raise InputError(f'PCE basis exceeds {MAX_PCE_TERMS} terms')
    return out


def _pce_standardize(X:np.ndarray,marginals:Sequence[Mapping[str,Any]]):
    if len(marginals)!=X.shape[1]:raise InputError('one PCE marginal required per input')
    Z=np.empty_like(X,dtype=float);kinds=[]
    for j,s in enumerate(marginals):
        kind=str(s.get('kind','')).lower()
        if kind in {'normal','gaussian'}:
            mu=float(s.get('mean',s.get('loc',0)));sd=float(s.get('std',s.get('scale',1)))
            if sd<=0:raise InputError('normal PCE marginal needs sd>0')
            Z[:,j]=(X[:,j]-mu)/sd;kinds.append('normal')
        elif kind in {'uniform','rectangular'}:
            lo=float(s.get('lower'));hi=float(s.get('upper'))
            if not lo<hi:raise InputError('uniform PCE marginal needs lower<upper')
            Z[:,j]=2*(X[:,j]-lo)/(hi-lo)-1;kinds.append('uniform')
        else:raise InputError('PCE currently supports independent normal and uniform marginals only')
    return Z,kinds


def _basis_1d(z:np.ndarray,degree:int,kind:str):
    if kind=='normal':return eval_hermitenorm(degree,z)/math.sqrt(float(factorial(degree,exact=False)))
    return math.sqrt(2*degree+1)*eval_legendre(degree,z)


def _design(Z:np.ndarray,kinds,indices):
    A=np.ones((Z.shape[0],len(indices)))
    for k,mi in enumerate(indices):
        for j,deg in enumerate(mi):
            if deg:A[:,k]*=_basis_1d(Z[:,j],deg,kinds[j])
    return A


def _pce_fit_raw(X,y,marginals,degree):
    if degree<1 or degree>8:raise InputError('PCE total_degree must lie in [1,8]')
    Z,kinds=_pce_standardize(X,marginals);inds=_multiindices(X.shape[1],degree);A=_design(Z,kinds,inds);coef,*_=np.linalg.lstsq(A,y,rcond=None)
    return {'coef':coef,'indices':inds,'kinds':kinds,'marginals':list(marginals),'degree':degree}


def _pce_predict(model,X):
    Z,k=_pce_standardize(np.asarray(X,dtype=float),model['marginals']);return _design(Z,k,model['indices'])@model['coef']


def polynomial_chaos(payload:Mapping[str,Any])->dict[str,Any]:
    X,y=_xy(payload);marg=list(payload.get('marginals') or []);degree=int(payload.get('total_degree',2));runs=[]
    for seed in _validation_seeds(payload):
        pp=dict(payload);pp['validation_seed']=seed;tr,va=_split(len(y),pp);mm=_pce_fit_raw(X[tr],y[tr],marg,degree);pred=_pce_predict(mm,X[va]);runs.append({'seed':seed,'n_train':int(len(tr)),'n_validation':int(len(va)),'metrics':_metrics(y[va],pred)})
    metrics=_aggregate_metrics([x['metrics'] for x in runs]);status,q=_qualify(metrics,payload);m=_pce_fit_raw(X,y,marg,degree)
    coef=np.asarray(m['coef']);mean=float(coef[0]);var=float(np.sum(coef[1:]**2))
    return {'engine_version':ENGINE_VERSION,'kind':'polynomial_chaos_orthonormal','status':status,'total_degree':degree,'basis_terms':len(coef),'validation_repeats':len(runs),
            'validation_metrics':metrics,'validation_runs':runs,'qualification':q,'distribution_mean_from_coefficients':mean,'distribution_variance_from_coefficients':var,
            'coefficients':coef.tolist(),'multiindices':[list(x) for x in m['indices']],
            'claim_boundary':'Coefficient moments assume the declared independent normal/uniform marginals and orthonormal basis; qualification uses the worst metric across deterministic held-out splits.'}


def propagate_gp(payload:Mapping[str,Any])->dict[str,Any]:
    X,y=_xy(payload);dist=dict(payload.get('input_distribution') or {});n=int(payload.get('n',100000));seed=int(payload.get('seed',1729));dist['n']=n;dist['seed']=seed
    # First obtain independent held-out validation on a training subset.
    val=gaussian_process(payload)
    if val['status']!='SURROGATE_QUALIFIED':return {**val,'propagation_status':'BLOCKED_UNQUALIFIED_SURROGATE'}
    model=_gp_fit_raw(X,y,payload);names,S,_=joint.generate_inputs(dist);mean,var=_gp_predict(model,S)
    physical=float(np.var(mean,ddof=1));surrogate=float(np.mean(var));total=physical+surrogate
    return {**val,'propagation_status':'PASS','input_variable_order':names,'n_propagation':int(len(mean)),'output_mean':float(np.mean(mean)),
            'variance_due_to_input_through_surrogate_mean':physical,'mean_GP_predictive_variance':surrogate,'total_predictive_variance_law_total_variance':total,
            'output_std_total_predictive':math.sqrt(max(0,total)),
            'claim_boundary_propagation':'GP predictive variance is reported separately from input-driven variance; it is not relabeled as measurement uncertainty.'}


def propagate_pce(payload:Mapping[str,Any])->dict[str,Any]:
    X,y=_xy(payload);val=polynomial_chaos(payload)
    if val['status']!='SURROGATE_QUALIFIED':return {**val,'propagation_status':'BLOCKED_UNQUALIFIED_SURROGATE'}
    marg=list(payload.get('marginals') or []);degree=int(payload.get('total_degree',2));m=_pce_fit_raw(X,y,marg,degree)
    return {**val,'propagation_status':'PASS','output_mean':float(m['coef'][0]),'output_variance':float(np.sum(np.asarray(m['coef'][1:])**2)),
            'output_std':math.sqrt(max(0,float(np.sum(np.asarray(m['coef'][1:])**2)))),
            'claim_boundary_propagation':'PCE moments describe the qualified surrogate under the declared input distribution; validation error is not silently added as a physical uncertainty component.'}
