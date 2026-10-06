#!/usr/bin/env python3
"""Global and nonlinear Bayesian experimental design for Uncertainty Lab v0.9.

Routes:
- exact exhaustive small-batch search for the established linear-Gaussian design;
- sample-based expected information gain (EIG) for nonlinear scalar candidate
  measurements with Gaussian observation noise;
- exhaustive small-batch nonlinear EIG for conditionally independent candidate
  measurements.

EIG is estimated by nested Monte Carlo over an explicit parameter-particle
representation. Estimator replication error is reported; finite-particle EIG is
not presented as an exact mutual information for the unknown continuous posterior.
"""
from __future__ import annotations
import itertools, math
from typing import Any, Mapping
import numpy as np
from scipy.special import logsumexp
import experimental_design_engine as lin
import joint_distribution_engine as joint
ENGINE_VERSION='0.9.0'
MAX_CANDIDATES_GLOBAL=20
MAX_PARTICLES=4000
MAX_OUTER=800
class InputError(ValueError): pass


def exact_global_batch(payload:Mapping[str,Any])->dict[str,Any]:
    m,P=lin._state(payload); rows=lin._candidates(payload,len(m)); k=int(payload.get('batch_size',2)); objective=str(payload.get('objective','information_gain')).lower(); per_cost=bool(payload.get('per_cost',False))
    if len(rows)>MAX_CANDIDATES_GLOBAL or not 1<=k<=min(5,len(rows)): raise InputError('global batch exact search supports <=20 candidates and batch_size<=5')
    if objective not in {'information_gain','trace_reduction','target_variance_reduction'}: raise InputError('global exact batch objective must be information_gain, trace_reduction, or target_variance_reduction')
    c=None
    if objective=='target_variance_reduction':
        c=np.asarray(payload.get('target_sensitivity'),dtype=float)
        if c.shape!=(len(m),):raise InputError('target_sensitivity required for target objective')
    prior_ld=lin._logdet(P); prior_trace=float(np.trace(P)); prior_tv=None if c is None else float(c@P@c)
    scored=[]
    for combo in itertools.combinations(rows,k):
        Q=P.copy(); cost=0.0
        for r in combo:
            Q,_,_=lin._update(Q,r['h'],r['noise_var']); cost+=r['cost']
        ig=.5*(prior_ld-lin._logdet(Q)); tr=prior_trace-float(np.trace(Q)); tv=None if c is None else prior_tv-float(c@Q@c)
        raw={'information_gain':ig,'trace_reduction':tr,'target_variance_reduction':tv}[objective]; score=float(raw)/(cost if per_cost else 1.0)
        scored.append({'candidates':[r['name'] for r in combo],'score':score,'raw_objective_value':float(raw),'total_cost':cost,'posterior_covariance':Q.tolist(),'information_gain_nats':ig,'trace_reduction':tr,'target_variance_reduction':tv})
    scored.sort(key=lambda x:x['score'],reverse=True)
    return {'engine_version':ENGINE_VERSION,'kind':'global_linear_gaussian_batch_design','objective':objective,'per_cost':per_cost,'evaluated_combinations':len(scored),'best_batch':scored[0],'ranked_batches':scored[:int(payload.get('top_k',10))],
            'claim_boundary':'Global optimum is exhaustive only over the finite candidate set and requested batch size under the declared linear-Gaussian model.'}


def _particles(payload):
    names=list(payload.get('parameter_names') or []); X=np.asarray(payload.get('parameter_samples'),dtype=float)
    if X.ndim!=2 or X.shape[0]<50 or X.shape[0]>MAX_PARTICLES or X.shape[1]!=len(names) or len(set(names))!=len(names) or not np.isfinite(X).all(): raise InputError('parameter_samples/parameter_names invalid')
    w=payload.get('weights')
    if w is None: w=np.full(len(X),1/len(X))
    else:
        w=np.asarray(w,dtype=float)
        if w.shape!=(len(X),) or np.any(w<0) or not np.isfinite(w).all() or w.sum()<=0: raise InputError('weights invalid')
        w=w/w.sum()
    return names,X,w


def _candidate_predictions(candidate,names,X):
    expr=str(candidate.get('model_expression','')).strip(); sd=float(candidate.get('noise_std')); cost=float(candidate.get('cost',1.0)); name=str(candidate.get('name','')).strip()
    if not name or not expr or not math.isfinite(sd) or sd<=0 or not math.isfinite(cost) or cost<=0: raise InputError('invalid nonlinear design candidate')
    node=joint._compile(expr); env={n:X[:,i] for i,n in enumerate(names)}
    data=dict(candidate.get('data') or {})
    for k,v in data.items():
        a=np.asarray(v,dtype=float)
        if a.ndim==0: env[k]=float(a)
        elif a.shape==(len(X),): env[k]=a
        else: raise InputError('candidate data must be scalar or one value per parameter particle')
    with np.errstate(all='ignore'): mu=np.asarray(joint._eval(node,env),dtype=float)
    if mu.ndim==0: mu=np.full(len(X),float(mu))
    if mu.shape!=(len(X),) or not np.isfinite(mu).all(): raise InputError('candidate model expression invalid on parameter samples')
    return name,mu,sd,cost


def _eig_one(mu,sd,w,outer,seed):
    rng=np.random.Generator(np.random.PCG64(seed)); idx=rng.choice(len(mu),size=outer,p=w); y=mu[idx]+sd*rng.normal(size=outer)
    # log p(y|theta_i)
    log_cond=-.5*((y-mu[idx])/sd)**2-math.log(sd)-.5*math.log(2*math.pi)
    # finite-particle mixture evidence p(y)
    z=(y[:,None]-mu[None,:])/sd; log_components=-.5*z*z-math.log(sd)-.5*math.log(2*math.pi)+np.log(w)[None,:]
    log_marg=logsumexp(log_components,axis=1)
    vals=log_cond-log_marg
    return float(vals.mean()), float(vals.std(ddof=1)/math.sqrt(outer))


def nonlinear_eig(payload:Mapping[str,Any])->dict[str,Any]:
    names,X,w=_particles(payload); candidates=list(payload.get('candidates') or []); outer=int(payload.get('outer_samples',400)); reps=int(payload.get('replicates',4)); seed=int(payload.get('seed',1729)); per_cost=bool(payload.get('per_cost',False))
    if not 1<=len(candidates)<=100 or not 100<=outer<=MAX_OUTER or not 2<=reps<=12: raise InputError('invalid nonlinear EIG settings')
    rows=[]
    for j,c in enumerate(candidates):
        name,mu,sd,cost=_candidate_predictions(c,names,X); vals=[]
        for r in range(reps): vals.append(_eig_one(mu,sd,w,outer,seed+j*100003+r*7919)[0])
        mean=float(np.mean(vals)); se=float(np.std(vals,ddof=1)/math.sqrt(reps)); score=mean/(cost if per_cost else 1.0)
        rows.append({'name':name,'expected_information_gain_nats':mean,'replicate_standard_error':se,'cost':cost,'score':score,'replicate_values':vals})
    rows.sort(key=lambda z:z['score'],reverse=True)
    return {'engine_version':ENGINE_VERSION,'kind':'nonlinear_particle_expected_information_gain','status':'PASS','per_cost':per_cost,'parameter_particles':len(X),'outer_samples_per_replicate':outer,'replicates':reps,'ranked_candidates':rows,'best_candidate':rows[0],
            'claim_boundary':'EIG is a nested Monte Carlo estimate under the finite supplied parameter-particle distribution and Gaussian candidate likelihoods; replicate SE measures estimator variability, not model-form uncertainty.'}


def _eig_combo(preds,sds,w,outer,seed):
    rng=np.random.Generator(np.random.PCG64(seed)); N=preds.shape[1]; idx=rng.choice(N,size=outer,p=w); k=preds.shape[0]; Y=preds[:,idx].T+rng.normal(size=(outer,k))*sds[None,:]
    # conditionally independent Gaussian candidate measurements
    logc=np.sum(-.5*((Y-preds[:,idx].T)/sds)**2-np.log(sds)-.5*math.log(2*math.pi),axis=1)
    logmix=np.empty((outer,N))
    for o in range(outer):
        z=(Y[o,:,None]-preds)/sds[:,None]; logmix[o]=np.sum(-.5*z*z-np.log(sds)[:,None]-.5*math.log(2*math.pi),axis=0)+np.log(w)
    vals=logc-logsumexp(logmix,axis=1); return float(vals.mean())


def nonlinear_global_batch(payload:Mapping[str,Any])->dict[str,Any]:
    names,X,w=_particles(payload); cand=list(payload.get('candidates') or []); k=int(payload.get('batch_size',2)); outer=int(payload.get('outer_samples',250)); reps=int(payload.get('replicates',3)); seed=int(payload.get('seed',1729)); per_cost=bool(payload.get('per_cost',False))
    if len(cand)>15 or not 1<=k<=min(3,len(cand)) or not 100<=outer<=500 or not 2<=reps<=8: raise InputError('nonlinear global batch limits exceeded')
    parsed=[_candidate_predictions(c,names,X) for c in cand]; scored=[]
    for ci,combo in enumerate(itertools.combinations(range(len(parsed)),k)):
        preds=np.vstack([parsed[i][1] for i in combo]); sds=np.array([parsed[i][2] for i in combo]); cost=sum(parsed[i][3] for i in combo); vals=[_eig_combo(preds,sds,w,outer,seed+ci*10007+r*7919) for r in range(reps)]; mean=float(np.mean(vals)); se=float(np.std(vals,ddof=1)/math.sqrt(reps)); score=mean/(cost if per_cost else 1.0)
        scored.append({'candidates':[parsed[i][0] for i in combo],'expected_information_gain_nats':mean,'replicate_standard_error':se,'total_cost':cost,'score':score})
    scored.sort(key=lambda z:z['score'],reverse=True)
    return {'engine_version':ENGINE_VERSION,'kind':'global_nonlinear_particle_batch_eig','status':'PASS','evaluated_combinations':len(scored),'best_batch':scored[0],'ranked_batches':scored[:int(payload.get('top_k',10))],
            'claim_boundary':'Global optimum is exhaustive over the finite candidate combinations evaluated; EIG remains a nested Monte Carlo estimate under the supplied finite particle posterior.'}
