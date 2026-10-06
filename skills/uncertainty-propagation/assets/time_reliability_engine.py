#!/usr/bin/env python3
"""Time-dependent reliability and first-passage analysis for v0.9.

Qualified routes:
- Brownian motion with constant drift/diffusion, including Brownian-bridge
  within-step crossing correction and exact closed-form benchmark probability.
- Stationary Ornstein-Uhlenbeck process on an explicit time grid.
- Discrete stationary AR(1) process.

OU/AR1 results are grid-event probabilities unless otherwise stated; the engine
never silently upgrades discrete monitoring to continuous-time first-passage.
"""
from __future__ import annotations
import math
from typing import Any, Mapping
import numpy as np
from scipy import stats

ENGINE_VERSION='0.9.0'
MAX_PATH_VALUES=20_000_000
class InputError(ValueError): pass


def _side(payload):
    side=str(payload.get('failure_side','upper')).lower()
    if side not in {'upper','lower'}: raise InputError('failure_side must be upper or lower')
    return side


def _summarize(hit, first_idx, times):
    n=len(hit); p=float(np.mean(hit)); se=math.sqrt(max(p*(1-p),0)/n)
    ft=times[first_idx[hit]] if np.any(hit) else np.array([])
    return {'failure_probability':p,'standard_error':se,'failure_count':int(np.sum(hit)),'n':n,
            'first_passage_time_mean_conditional':None if ft.size==0 else float(ft.mean()),
            'first_passage_time_quantiles_conditional':None if ft.size==0 else [float(x) for x in np.quantile(ft,[.05,.5,.95])]}


def brownian(payload:Mapping[str,Any])->dict[str,Any]:
    x0=float(payload.get('x0',0)); mu=float(payload.get('drift',0)); sigma=float(payload.get('sigma',1)); barrier=float(payload.get('barrier')); T=float(payload.get('horizon',1)); steps=int(payload.get('steps',256)); n=int(payload.get('n',50000)); seed=int(payload.get('seed',1729)); side=_side(payload)
    if not all(map(math.isfinite,[x0,mu,sigma,barrier,T])) or sigma<=0 or T<=0 or steps<2 or n<1000 or n*(steps+1)>MAX_PATH_VALUES: raise InputError('invalid/oversized Brownian settings')
    # Map lower crossing to upper crossing by sign flip.
    if side=='lower': x0,mu,barrier=-x0,-mu,-barrier
    if x0>=barrier:
        return {'engine_version':ENGINE_VERSION,'kind':'brownian_first_passage','status':'PASS','failure_probability':1.0,'standard_error':0.0,'n':n,'exact_failure_probability':1.0,'bridge_correction':True}
    rng=np.random.Generator(np.random.PCG64(seed)); dt=T/steps; sd=sigma*math.sqrt(dt); times=np.linspace(0,T,steps+1)
    x=np.full(n,x0); hit=np.zeros(n,dtype=bool); first=np.full(n,steps,dtype=int)
    for k in range(1,steps+1):
        prev=x.copy(); x=prev+mu*dt+sd*rng.normal(size=n)
        direct=(x>=barrier)&(~hit)
        first[direct]=k; hit|=direct
        # Conditional Brownian bridge crossing between two below-barrier endpoints.
        cand=(~hit)&(prev<barrier)&(x<barrier)
        if np.any(cand):
            pc=np.exp(-2*(barrier-prev[cand])*(barrier-x[cand])/(sigma*sigma*dt)); cross=rng.random(np.sum(cand))<pc
            idx=np.flatnonzero(cand)[cross]; first[idx]=k; hit[idx]=True
    out=_summarize(hit,first,times)
    a=barrier-x0; z1=(mu*T-a)/(sigma*math.sqrt(T)); z2=(-mu*T-a)/(sigma*math.sqrt(T)); exact=float(stats.norm.cdf(z1)+math.exp(2*mu*a/(sigma*sigma))*stats.norm.cdf(z2))
    out.update({'engine_version':ENGINE_VERSION,'kind':'brownian_first_passage','status':'PASS','horizon':T,'steps':steps,'bridge_correction':True,'exact_failure_probability':min(1.0,max(0.0,exact)),
                'claim_boundary':'Brownian-bridge correction targets continuous barrier crossing for constant-drift Brownian motion; reported Monte Carlo SE is simulation uncertainty.'})
    return out


def ou(payload:Mapping[str,Any])->dict[str,Any]:
    mean=float(payload.get('long_run_mean',0)); theta=float(payload.get('mean_reversion')); stationary_sd=float(payload.get('stationary_std')); x0=float(payload.get('x0',mean)); barrier=float(payload.get('barrier')); T=float(payload.get('horizon',1)); steps=int(payload.get('steps',200)); n=int(payload.get('n',50000)); seed=int(payload.get('seed',1729)); side=_side(payload)
    if theta<=0 or stationary_sd<=0 or T<=0 or steps<2 or n<1000 or n*(steps+1)>MAX_PATH_VALUES: raise InputError('invalid/oversized OU settings')
    if side=='lower': mean,x0,barrier=-mean,-x0,-barrier
    dt=T/steps; phi=math.exp(-theta*dt); trans_sd=stationary_sd*math.sqrt(1-phi*phi); rng=np.random.Generator(np.random.PCG64(seed)); times=np.linspace(0,T,steps+1)
    x=np.full(n,x0); hit=x>=barrier; first=np.zeros(n,dtype=int); first[~hit]=steps
    for k in range(1,steps+1):
        x=mean+phi*(x-mean)+trans_sd*rng.normal(size=n); new=(x>=barrier)&(~hit); first[new]=k; hit|=new
    out=_summarize(hit,first,times); out.update({'engine_version':ENGINE_VERSION,'kind':'ou_grid_first_passage','status':'PASS','horizon':T,'steps':steps,'grid_dt':dt,
        'claim_boundary':'OU failure probability is for the explicit monitoring grid. Continuous crossings between grid points are not inferred.'}); return out


def ar1(payload:Mapping[str,Any])->dict[str,Any]:
    phi=float(payload.get('phi')); mean=float(payload.get('mean',0)); stationary_sd=float(payload.get('stationary_std')); x0=float(payload.get('x0',mean)); barrier=float(payload.get('barrier')); steps=int(payload.get('steps')); n=int(payload.get('n',50000)); seed=int(payload.get('seed',1729)); side=_side(payload)
    if abs(phi)>=1 or stationary_sd<=0 or steps<1 or n<1000 or n*(steps+1)>MAX_PATH_VALUES: raise InputError('invalid/oversized AR1 settings')
    if side=='lower': mean,x0,barrier=-mean,-x0,-barrier
    innovation_sd=stationary_sd*math.sqrt(1-phi*phi); rng=np.random.Generator(np.random.PCG64(seed)); x=np.full(n,x0); hit=x>=barrier; first=np.zeros(n,dtype=int); first[~hit]=steps
    for k in range(1,steps+1):
        x=mean+phi*(x-mean)+innovation_sd*rng.normal(size=n); new=(x>=barrier)&(~hit); first[new]=k; hit|=new
    out=_summarize(hit,first,np.arange(steps+1,dtype=float)); out.update({'engine_version':ENGINE_VERSION,'kind':'ar1_discrete_first_passage','status':'PASS','steps':steps,'claim_boundary':'AR(1) reliability is a discrete-time event probability.'}); return out
