#!/usr/bin/env python3
"""Nonlinear variance attribution with explicit dependence semantics.

Independent inputs use Sobol first/total indices. Correlated Gaussian inputs use
variance-game Shapley effects estimated from conditional Gaussian distributions.
Correlated inputs are never fed to ordinary Sobol formulae.
"""
from __future__ import annotations
import itertools, math
from typing import Any, Mapping, Sequence
import numpy as np
import joint_distribution_engine as joint
ENGINE_VERSION='0.5.0'
MAX_SHAPLEY_DIM=6
class InputError(ValueError):pass

def _eval_expr(expression:str,names:Sequence[str],X:np.ndarray)->np.ndarray:
    node=joint._compile(expression);env={n:X[:,i] for i,n in enumerate(names)}
    with np.errstate(all='ignore'): y=np.asarray(joint._eval(node,env),dtype=float)
    if y.ndim==0:y=np.full(X.shape[0],float(y))
    if y.shape!=(X.shape[0],) or not np.isfinite(y).all():raise InputError('sensitivity evaluation produced invalid/non-finite values; conditional attribution is not silently substituted')
    return y

def _sample_independent(marginals, n, rng):
    U=rng.random((n,len(marginals)))
    return np.column_stack([joint._marginal_ppf(U[:,i],marginals[i]) for i in range(len(marginals))])

def _sobol_once(names,marginals,expression,n,seed):
    rng=np.random.Generator(np.random.PCG64(seed));A=_sample_independent(marginals,n,rng);B=_sample_independent(marginals,n,rng)
    fA=_eval_expr(expression,names,A);fB=_eval_expr(expression,names,B);V=float(np.var(np.r_[fA,fB],ddof=1))
    if V<=0:raise InputError('output variance is zero; Sobol indices undefined')
    first=[];total=[]
    for i in range(len(names)):
        C=A.copy();C[:,i]=B[:,i];fC=_eval_expr(expression,names,C)
        first.append(float(np.mean(fB*(fC-fA))/V))
        total.append(float(np.mean((fA-fC)**2)/(2*V)))
    return np.array(first),np.array(total),V

def sobol_independent(payload:Mapping[str,Any])->dict[str,Any]:
    names=list(payload.get('variables') or []);marg=list(payload.get('marginals') or []);ex=str(payload.get('expression',''))
    n=int(payload.get('n',100000));reps=int(payload.get('replicates',3));seed=int(payload.get('seed',1729))
    if not names or len(names)!=len(marg) or len(set(names))!=len(names):raise InputError('variables/marginals mismatch')
    if n<1000 or n>1_000_000 or reps<1 or reps>20:raise InputError('invalid n/replicates')
    Fs=[];Ts=[];Vs=[]
    for r in range(reps):
        f,t,v=_sobol_once(names,marg,ex,n,seed+r*104729);Fs.append(f);Ts.append(t);Vs.append(v)
    F=np.vstack(Fs);T=np.vstack(Ts)
    return {'engine_version':ENGINE_VERSION,'kind':'sobol_independent','variable_order':names,'n_per_replicate':n,'replicates':reps,'seed':seed,
            'first_order_mean':F.mean(axis=0).tolist(),'first_order_sd_across_replicates':F.std(axis=0,ddof=1).tolist() if reps>1 else [None]*len(names),
            'total_order_mean':T.mean(axis=0).tolist(),'total_order_sd_across_replicates':T.std(axis=0,ddof=1).tolist() if reps>1 else [None]*len(names),
            'output_variance_mean':float(np.mean(Vs)),'assumption':'Inputs are mutually independent; Sobol indices are not used for correlated inputs.'}

def _validate_gaussian(mean,cov,names):
    mu=np.asarray(mean,dtype=float);C=np.asarray(cov,dtype=float);d=len(names)
    if mu.shape!=(d,) or C.shape!=(d,d) or not np.isfinite(mu).all() or not np.isfinite(C).all():raise InputError('mean/covariance mismatch')
    diag=np.diag(C)
    if np.any(diag<=0):raise InputError('Gaussian Shapley currently requires positive variances')
    sd=np.sqrt(diag);R=C/np.outer(sd,sd);scale=max(1.0,float(np.max(np.abs(R))))
    if np.max(np.abs(R-R.T))>1e-12+1e-10*scale:raise InputError('covariance not symmetric on variance-normalized basis')
    R=(R+R.T)/2;ev=np.linalg.eigvalsh(R);tol=1e-12+1e-10*max(1,float(np.max(np.abs(ev))))
    if ev[0]<-tol:raise InputError('covariance not PSD')
    return mu,sd,R

def _corr_sample(R,n,rng):
    ev,Q=np.linalg.eigh((R+R.T)/2);tol=1e-12+1e-10*max(1.0,float(np.max(np.abs(ev))));keep=ev>tol
    if not np.any(keep):return np.zeros((n,R.shape[0]))
    return rng.normal(size=(n,int(np.sum(keep))))@(Q[:,keep]*np.sqrt(ev[keep])).T

def _game_value(mask:int,names,mu,sd,R,expression,outer,inner,seed):
    d=len(names);S=[i for i in range(d) if mask>>i&1];T=[i for i in range(d) if not(mask>>i&1)]
    rng=np.random.Generator(np.random.PCG64(seed))
    if not S:return 0.0
    if not T:
        Z=_corr_sample(R,outer,rng);X=mu+Z*sd;return float(np.var(_eval_expr(expression,names,X),ddof=1))
    SS=R[np.ix_(S,S)];TS=R[np.ix_(T,S)];TT=R[np.ix_(T,T)]
    invSS=np.linalg.pinv(SS,rcond=1e-12);A=TS@invSS;cond=TT-A@R[np.ix_(S,T)];cond=(cond+cond.T)/2
    ZS=_corr_sample(SS,outer,rng);means=np.empty(outer)
    for k,zs in enumerate(ZS):
        cm=A@zs;ZT=cm+_corr_sample(cond,inner,rng);Z=np.empty((inner,d));Z[:,S]=zs;Z[:,T]=ZT;X=mu+Z*sd;means[k]=_eval_expr(expression,names,X).mean()
    return float(np.var(means,ddof=1))

def _shapley_once(names,mu,sd,R,expression,outer,inner,seed):
    d=len(names);full=(1<<d)-1;game={}
    for mask in range(1<<d):game[mask]=_game_value(mask,names,mu,sd,R,expression,outer,inner,seed+mask*7919)
    total=game[full]
    if total<=0:raise InputError('output variance is zero; Shapley effects undefined')
    phi=np.zeros(d);fact=math.factorial
    for i in range(d):
        bit=1<<i
        for mask in range(1<<d):
            if mask&bit:continue
            ss=mask.bit_count();w=fact(ss)*fact(d-ss-1)/fact(d)
            phi[i]+=w*(game[mask|bit]-game[mask])
    return phi/total,total,game

def shapley_correlated_gaussian(payload:Mapping[str,Any])->dict[str,Any]:
    names=list(payload.get('variables') or []);d=len(names)
    if not names or len(set(names))!=d or d>MAX_SHAPLEY_DIM:raise InputError(f'Gaussian Shapley requires 1..{MAX_SHAPLEY_DIM} unique variables')
    mu,sd,R=_validate_gaussian(payload.get('mean'),payload.get('covariance'),names);ex=str(payload.get('expression',''))
    outer=int(payload.get('outer_samples',1000));inner=int(payload.get('inner_samples',200));reps=int(payload.get('replicates',2));seed=int(payload.get('seed',1729))
    if outer<100 or inner<20 or reps<1 or reps>10:raise InputError('insufficient/invalid Monte Carlo settings')
    P=[];Vs=[]
    for r in range(reps):
        p,v,_=_shapley_once(names,mu,sd,R,ex,outer,inner,seed+r*1000003);P.append(p);Vs.append(v)
    P=np.vstack(P)
    return {'engine_version':ENGINE_VERSION,'kind':'shapley_correlated_gaussian','variable_order':names,'effects_mean':P.mean(axis=0).tolist(),
            'effects_sd_across_replicates':P.std(axis=0,ddof=1).tolist() if reps>1 else [None]*d,'effects_sum':float(P.mean(axis=0).sum()),
            'output_variance_mean':float(np.mean(Vs)),'outer_samples':outer,'inner_samples':inner,'replicates':reps,'seed':seed,
            'assumption':'Joint inputs are multivariate Gaussian. Effects use the conditional-expectation variance game and retain dependence rather than applying independent-input Sobol formulae.'}
