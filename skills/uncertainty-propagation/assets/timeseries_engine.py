#!/usr/bin/env python3
"""Explicit repeatability/time-series uncertainty models.

No model is selected automatically.  The caller must request IID, AR(1), HAC,
circular block bootstrap, or balanced random-effects.  Diagnostics never silently change
the requested estimator.
"""
from __future__ import annotations
from typing import Any, Mapping
import math
import numpy as np
from scipy import stats
ENGINE_VERSION='0.5.0'
MAX_OBSERVATIONS=2_000_000
class InputError(ValueError):pass

def _series(x:Any,min_n=3)->np.ndarray:
    a=np.asarray(x,dtype=float)
    if a.ndim!=1 or len(a)<min_n or len(a)>MAX_OBSERVATIONS or not np.isfinite(a).all():raise InputError(f'observations must be finite 1-D with {min_n}<=n<={MAX_OBSERVATIONS}')
    return a

def diagnostics(x:Any)->dict[str,Any]:
    a=_series(x);n=len(a);c=a-a.mean();den=float(c@c);lag1=float(c[1:]@c[:-1]/den) if den>0 else None
    t=np.arange(n,dtype=float);slope=float(np.polyfit(t,a,1)[0]) if n>=3 else None
    first=float(np.mean(a[:n//2]));second=float(np.mean(a[n//2:]))
    return {'n':n,'mean':float(a.mean()),'sample_std':float(a.std(ddof=1)),'lag1_autocorrelation':lag1,
            'linear_drift_per_index':slope,'first_half_mean':first,'second_half_mean':second,'half_mean_difference':second-first,
            'note':'Diagnostics are evidence only; they do not select or replace the explicit dependence model.'}

def iid_mean(x:Any)->dict[str,Any]:
    a=_series(x);n=len(a);s=float(a.std(ddof=1))
    return {'engine_version':ENGINE_VERSION,'kind':'iid_mean','estimate':float(a.mean()),'standard_uncertainty':s/math.sqrt(n),'degrees_of_freedom':n-1,
            'assumption':'Observations are independent and identically distributed with finite variance.','diagnostics':diagnostics(a)}

def ar1_mean(x:Any,*,phi:float|None=None,estimate_phi:bool=False)->dict[str,Any]:
    a=_series(x,5);n=len(a);c=a-a.mean()
    if phi is None:
        if not estimate_phi:raise InputError('AR(1) requires explicit phi or estimate_phi=true')
        den=float(c[:-1]@c[:-1]);
        if den==0:raise InputError('cannot estimate AR(1) phi from constant series')
        phi=float(c[1:]@c[:-1]/den);phi_source='estimated_OLS_on_demeaned_series'
    else:phi=float(phi);phi_source='supplied'
    if not math.isfinite(phi) or abs(phi)>=1:raise InputError('stationary AR(1) requires |phi|<1')
    resid=c[1:]-phi*c[:-1];df=max(1,len(resid)-(1 if estimate_phi and phi_source.startswith('estimated') else 0));sigma_e2=float(resid@resid/df)
    gamma0=sigma_e2/(1-phi*phi)
    k=np.arange(1,n,dtype=float);factor=n+2*float(np.sum((n-k)*(phi**k)))
    var_mean=max(0.0,gamma0*factor/(n*n))
    return {'engine_version':ENGINE_VERSION,'kind':'ar1_mean','estimate':float(a.mean()),'standard_uncertainty':math.sqrt(var_mean),'phi':phi,'phi_source':phi_source,
            'innovation_variance':sigma_e2,'marginal_variance_model':gamma0,'variance_mean_finite_n':var_mean,
            'assumption':'Stationary AR(1) residual process around a constant mean.','diagnostics':diagnostics(a)}

def hac_mean(x:Any,*,bandwidth:int)->dict[str,Any]:
    a=_series(x);n=len(a);L=int(bandwidth)
    if L<0 or L>=n:raise InputError('HAC bandwidth must satisfy 0<=L<n')
    c=a-a.mean();gamma0=float(c@c/n);longrun=gamma0;accum_scale=abs(gamma0);gammas=[]
    for k in range(1,L+1):
        g=float(c[k:]@c[:-k]/n);w=1-k/(L+1);term=2*w*g;longrun+=term;accum_scale+=abs(term);gammas.append({'lag':k,'gamma':g,'bartlett_weight':w})
    # Roundoff tolerance scales with the actual variance terms; no fixed
    # physical-unit floor is allowed.
    tol=64*np.finfo(float).eps*accum_scale
    if longrun < -tol:raise InputError('HAC long-run variance estimate is negative; chosen bandwidth/model is not numerically defensible')
    longrun=max(0.0,longrun);var_mean=longrun/n
    return {'engine_version':ENGINE_VERSION,'kind':'newey_west_hac_mean','estimate':float(a.mean()),'standard_uncertainty':math.sqrt(var_mean),'bandwidth':L,
            'long_run_variance':longrun,'variance_mean':var_mean,'kernel':'Bartlett','lag_terms':gammas,
            'assumption':'Weakly dependent stationary process; bandwidth was selected explicitly by the caller.','diagnostics':diagnostics(a)}

def circular_block_bootstrap_mean(x:Any,*,block_length:int,resamples:int=5000,seed:int=1729,probability:float=0.95)->dict[str,Any]:
    a=_series(x);n=len(a);L=int(block_length);B=int(resamples);p=float(probability)
    if L<1 or L>n:raise InputError('block_length must be in [1,n]')
    if B<200 or B>200000:raise InputError('resamples must be in [200,200000]')
    if not 0<p<1:raise InputError('probability must be in (0,1)')
    rng=np.random.Generator(np.random.PCG64(int(seed)));means=np.empty(B);blocks=math.ceil(n/L);base=np.arange(L)
    for b in range(B):
        starts=rng.integers(0,n,size=blocks);idx=((starts[:,None]+base[None,:])%n).reshape(-1)[:n];means[b]=a[idx].mean()
    lo=(1-p)/2;hi=1-lo
    return {'engine_version':ENGINE_VERSION,'kind':'circular_block_bootstrap_mean','estimate':float(a.mean()),'standard_uncertainty':float(means.std(ddof=1)),
            'block_length':L,'resamples':B,'seed':int(seed),'coverage_interval':{'probability':p,'method':'percentile','lower':float(np.quantile(means,lo)),'upper':float(np.quantile(means,hi))},
            'assumption':'Circular moving blocks approximate the dependence structure; block length was explicitly selected.','diagnostics':diagnostics(a)}

def balanced_random_effects(groups:Any)->dict[str,Any]:
    G=np.asarray(groups,dtype=float)
    if G.ndim!=2 or G.shape[0]<2 or G.shape[1]<2 or G.size>MAX_OBSERVATIONS or not np.isfinite(G).all():raise InputError('balanced random effects requires bounded finite rectangular groups with >=2 groups and >=2 repeats')
    g,m=G.shape;means=G.mean(axis=1);grand=float(G.mean());msb=float(m*np.var(means,ddof=1));msw=float(np.sum((G-means[:,None])**2)/(g*(m-1)))
    tau2=max(0.0,(msb-msw)/m);sigma2=msw;var_mean=tau2/g+sigma2/(g*m)
    return {'engine_version':ENGINE_VERSION,'kind':'balanced_random_intercept','estimate':grand,'standard_uncertainty':math.sqrt(var_mean),'groups':g,'repeats_per_group':m,
            'between_group_variance':tau2,'within_group_variance':sigma2,'variance_grand_mean':var_mean,'ms_between':msb,'ms_within':msw,
            'assumption':'Balanced one-way random-intercept model with independent groups and homoscedastic within-group noise.'}

def analyze(payload:Mapping[str,Any])->dict[str,Any]:
    kind=str(payload.get('kind','')).lower();x=payload.get('observations')
    if kind=='iid':return iid_mean(x)
    if kind=='ar1':return ar1_mean(x,phi=payload.get('phi'),estimate_phi=bool(payload.get('estimate_phi',False)))
    if kind=='hac':return hac_mean(x,bandwidth=int(payload['bandwidth']))
    if kind in {'block_bootstrap','circular_block_bootstrap'}:return circular_block_bootstrap_mean(x,block_length=int(payload['block_length']),resamples=int(payload.get('resamples',5000)),seed=int(payload.get('seed',1729)),probability=float(payload.get('probability',.95)))
    if kind in {'random_effects','balanced_random_effects'}:return balanced_random_effects(payload.get('groups'))
    raise InputError('explicit kind required: iid, ar1, hac, block_bootstrap, balanced_random_effects')
