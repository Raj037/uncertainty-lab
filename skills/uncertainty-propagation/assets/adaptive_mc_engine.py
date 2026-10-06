#!/usr/bin/env python3
"""Convergence-qualified Monte Carlo propagation for Uncertainty Lab v0.6.

Sampling continues in independent deterministic PCG64 batches until Monte Carlo numerical
uncertainty for requested summaries is below explicit tolerances, or a hard sample ceiling is
reached. Convergence is evidence about simulation precision, not scientific model validity.
"""
from __future__ import annotations
import math
from typing import Any, Mapping
import numpy as np
import joint_distribution_engine as joint

ENGINE_VERSION='0.6.0'
MAX_SAMPLES=2_000_000
MAX_BATCHES=256
class InputError(ValueError): pass

def _eval_batch(payload:Mapping[str,Any], n:int, seed:int):
    p=dict(payload); p['n']=int(n); p['seed']=int(seed)
    jm=dict(p.get('joint_model') or {}); jm['n']=int(n); jm['seed']=int(seed); p['joint_model']=jm
    names,X,meta=joint.generate_inputs(p)
    exprs=list(p.get('expressions') or ([p['expression']] if 'expression' in p else []))
    if not exprs: raise InputError('expression or expressions required')
    env={name:X[:,i] for i,name in enumerate(names)}; out=[]; valid=np.ones(n,dtype=bool)
    for e in exprs:
        c=joint._compile(e)
        with np.errstate(all='ignore'): y=np.asarray(joint._eval(c,env),dtype=float)
        if y.ndim==0:y=np.full(n,float(y))
        if y.shape!=(n,):raise InputError('expression did not produce one value per sample')
        valid &= np.isfinite(y); out.append(y)
    if not np.any(valid): return names,np.empty((len(exprs),0)),meta,n
    return names,np.vstack([y[valid] for y in out]),meta,int(np.sum(~valid))

def _summary(Y:np.ndarray, probability:float, batch_slices:list[tuple[int,int]]):
    n=Y.shape[1]; mean=Y.mean(axis=1); sd=Y.std(axis=1,ddof=1)
    lo=(1-probability)/2; hi=1-lo; qlo=np.quantile(Y,lo,axis=1); qhi=np.quantile(Y,hi,axis=1)
    # Independent-batch standard errors are distribution-agnostic. Use only batches with >=2 samples.
    bmeans=[]; bsds=[]; blos=[]; bhis=[]
    for a,b in batch_slices:
        if b-a<2: continue
        z=Y[:,a:b]; bmeans.append(z.mean(axis=1)); bsds.append(z.std(axis=1,ddof=1)); blos.append(np.quantile(z,lo,axis=1)); bhis.append(np.quantile(z,hi,axis=1))
    def se(rows):
        A=np.asarray(rows,dtype=float)
        if A.shape[0]<2:return np.full(Y.shape[0],np.inf)
        return A.std(axis=0,ddof=1)/math.sqrt(A.shape[0])
    return {'n_valid':n,'mean':mean,'std':sd,'qlo':qlo,'qhi':qhi,
            'se_mean':se(bmeans),'se_std':se(bsds),'se_qlo':se(blos),'se_qhi':se(bhis),'batch_count':len(bmeans)}

def _tol_ok(value,se,abs_tol,rel_tol,z):
    tol=abs_tol+rel_tol*np.maximum(np.abs(value),np.finfo(float).tiny)
    return np.asarray(z*se <= tol),tol

def propagate(payload:Mapping[str,Any])->dict[str,Any]:
    p=dict(payload)
    batch=int(p.get('batch_size',20_000)); min_n=int(p.get('min_samples',80_000)); max_n=int(p.get('max_samples',500_000)); base_seed=int(p.get('seed',1729))
    if batch<1000 or min_n<2*batch or max_n<min_n or max_n>MAX_SAMPLES: raise InputError('require batch_size>=1000, min_samples>=2*batch_size, min_samples<=max_samples<=2,000,000')
    prob=float(p.get('coverage_probability',.95));
    if not 0<prob<1:raise InputError('coverage_probability must be in (0,1)')
    abs_tol=float(p.get('mc_abs_tolerance',1e-3)); rel_tol=float(p.get('mc_rel_tolerance',5e-3)); z=float(p.get('mc_confidence_multiplier',2.0))
    if min(abs_tol,rel_tol)<0 or z<=0 or not all(map(math.isfinite,[abs_tol,rel_tol,z])):raise InputError('invalid Monte Carlo convergence tolerances')
    chunks=[]; slices=[]; invalid=0; total=0; meta=None; names=None; history=[]; converged=False; reason='MAX_SAMPLES_REACHED'
    # SeedSequence creates deterministic independent child streams, avoiding overlap between batches.
    children=np.random.SeedSequence(base_seed).spawn(min(MAX_BATCHES,math.ceil(max_n/batch)))
    for bi,ss in enumerate(children):
        if total>=max_n:break
        take=min(batch,max_n-total); seed=int(ss.generate_state(1,dtype=np.uint64)[0] % np.iinfo(np.int64).max)
        names,Y,m,inv=_eval_batch(p,take,seed); meta=m; invalid+=inv; start=sum(x.shape[1] for x in chunks); chunks.append(Y); end=start+Y.shape[1]; slices.append((start,end)); total+=take
        YY=np.concatenate(chunks,axis=1)
        if total<min_n or len(slices)<4:continue
        s=_summary(YY,prob,slices)
        oks=[]; diagnostics={}
        for key,sekey in [('mean','se_mean'),('std','se_std'),('qlo','se_qlo'),('qhi','se_qhi')]:
            ok,tol=_tol_ok(s[key],s[sekey],abs_tol,rel_tol,z); oks.append(ok); diagnostics[key]={'value':s[key].tolist(),'mc_standard_error':s[sekey].tolist(),'two_sided_numerical_halfwidth':(z*s[sekey]).tolist(),'tolerance':tol.tolist(),'passed':ok.tolist()}
        allok=bool(np.all(np.concatenate(oks)))
        history.append({'n_generated':total,'n_valid':s['n_valid'],'batches':s['batch_count'],'all_targets_pass':allok})
        if allok: converged=True;reason='CONVERGED';break
    YY=np.concatenate(chunks,axis=1)
    if YY.shape[1]<2:raise InputError('fewer than two valid Monte Carlo samples')
    s=_summary(YY,prob,slices)
    final={}
    all_flags=[]
    for key,sekey in [('mean','se_mean'),('std','se_std'),('qlo','se_qlo'),('qhi','se_qhi')]:
        ok,tol=_tol_ok(s[key],s[sekey],abs_tol,rel_tol,z);all_flags.append(ok);final[key]={'value':s[key].tolist(),'mc_standard_error':s[sekey].tolist(),'two_sided_numerical_halfwidth':(z*s[sekey]).tolist(),'tolerance':tol.tolist(),'passed':ok.tolist()}
    converged=bool(np.all(np.concatenate(all_flags))) and total>=min_n
    if converged:reason='CONVERGED'
    return {'engine_version':ENGINE_VERSION,'method':'sequential_independent_batch_MC','status':'CONVERGED' if converged else 'NOT_CONVERGED','reason':reason,'n_generated':total,'n_valid':int(YY.shape[1]),'invalid_count':invalid,'invalid_fraction':invalid/max(total,1),'batch_count':len(slices),'seed':base_seed,'rng':'NumPy PCG64 via SeedSequence child streams','coverage_probability':prob,'convergence':final,'history':history,'joint_model':meta,
            'mean':s['mean'].tolist(),'std':s['std'].tolist(),'equal_tailed_coverage_interval':{'probability':prob,'lower':s['qlo'].tolist(),'upper':s['qhi'].tolist()},
            'claim_boundary':'CONVERGED certifies Monte Carlo numerical precision for the reported summaries under the supplied joint model; it does not validate that scientific joint model.'}
