#!/usr/bin/env python3
"""Gradient-based Bayesian calibration for Uncertainty Lab v0.8.

Implements fixed-length HMC and the Hoffman-Gelman No-U-Turn Sampler (NUTS)
with dual-averaging step-size adaptation. The qualified v0.8 route deliberately
requires unconstrained normal priors; bounded/positive priors remain on the
existing Metropolis path until validated transforms are added.
"""
from __future__ import annotations
import math
from typing import Any, Mapping
import numpy as np
from scipy import linalg
import calibration_engine as cal
import joint_distribution_engine as joint
import posterior_diagnostics as postdiag

ENGINE_VERSION='0.8.0'
MAX_PARAMETERS=12
MAX_VALUES=2_000_000
class InputError(ValueError): pass

def _problem(payload:Mapping[str,Any]):
    names=list(payload.get('parameter_names') or []);d=len(names)
    if not 1<=d<=MAX_PARAMETERS or len(set(names))!=d:raise InputError(f'parameter_names must contain 1..{MAX_PARAMETERS} unique names')
    priors=dict(payload.get('priors') or {})
    if set(names)-set(priors):raise InputError('one explicit prior required per parameter')
    pm=np.empty(d);ps=np.empty(d)
    for i,n in enumerate(names):
        s=dict(priors[n]);k=str(s.get('kind','normal')).lower()
        if k not in {'normal','gaussian'}:raise InputError('qualified HMC/NUTS route currently requires unconstrained normal priors')
        pm[i]=float(s.get('mean',0));ps[i]=float(s.get('std',1))
        if not math.isfinite(pm[i]) or not math.isfinite(ps[i]) or ps[i]<=0:raise InputError(f'invalid normal prior for {n}')
    y=np.asarray(payload.get('observed'),dtype=float)
    if y.ndim!=1 or y.size<2 or not np.isfinite(y).all():raise InputError('observed must be finite vector')
    n=y.size;R,D,S=cal._obs_cov(payload,n);cf=linalg.cho_factor(S,lower=True);logdet=2*float(np.sum(np.log(np.diag(cf[0]))))
    data=dict(payload.get('data') or {})
    for k,v in list(data.items()):
        a=np.asarray(v,dtype=float)
        if a.shape!=(n,) or not np.isfinite(a).all():raise InputError(f'data[{k}] must be finite length n')
        data[k]=a
    expr=str(payload.get('model_expression','')).strip()
    if not expr:raise InputError('model_expression required')
    node=joint._compile(expr)
    def pred(q):
        env={**data,**{nm:float(q[i]) for i,nm in enumerate(names)}}
        with np.errstate(all='ignore'):p=np.asarray(joint._eval(node,env),dtype=float)
        if p.ndim==0:p=np.full(n,float(p))
        if p.shape!=(n,) or not np.isfinite(p).all():raise InputError('model expression produced invalid predictions')
        return p
    def logp(q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(d,) or not np.isfinite(q).all():return -math.inf
        z=(q-pm)/ps;lp=float(-.5*np.dot(z,z)-np.log(ps).sum()-.5*d*math.log(2*math.pi))
        try:r=y-pred(q)
        except Exception:return -math.inf
        quad=float(r@linalg.cho_solve(cf,r,check_finite=False))
        return lp-.5*(n*math.log(2*math.pi)+logdet+quad)
    return names,pm,ps,R,D,S,pred,logp

def _grad(logp,q):
    q=np.asarray(q,dtype=float);g=np.empty_like(q);eps=np.finfo(float).eps**.2
    for i,x in enumerate(q):
        h=eps*max(1.0,abs(float(x)));spacing=abs(float(np.spacing(float(x))));h=max(h,32*spacing,np.finfo(float).tiny)
        e=np.zeros_like(q);e[i]=h
        f2p,f1p,f1m,f2m=logp(q+2*e),logp(q+e),logp(q-e),logp(q-2*e)
        if not all(map(math.isfinite,[f2p,f1p,f1m,f2m])):raise InputError('non-finite posterior encountered during gradient evaluation')
        g[i]=(-f2p+8*f1p-8*f1m+f2m)/(12*h)
    return g

def _leapfrog(q,p,eps,logp):
    q=np.asarray(q,dtype=float).copy();p=np.asarray(p,dtype=float).copy();g=_grad(logp,q);p+=.5*eps*g;q+=eps*p;lp=logp(q)
    if not math.isfinite(lp):return q,p,-math.inf
    g=_grad(logp,q);p+=.5*eps*g;return q,p,lp

def _find_epsilon(q,logp,rng):
    p=rng.normal(size=len(q));lp=logp(q);q1,p1,lp1=_leapfrog(q,p,1.0,logp)
    if not math.isfinite(lp1):return .5
    loga=lp1-.5*np.dot(p1,p1)-(lp-.5*np.dot(p,p));a=1 if loga>math.log(.5) else -1;eps=1.0
    for _ in range(20):
        cond=(loga>math.log(.5)) if a==1 else (loga<math.log(.5))
        if not cond:break
        eps*=2.0**a
        if not 1e-6<=eps<=10:break
        q1,p1,lp1=_leapfrog(q,p,eps,logp)
        if not math.isfinite(lp1):loga=-math.inf
        else:loga=lp1-.5*np.dot(p1,p1)-(lp-.5*np.dot(p,p))
    return min(5.0,max(1e-5,eps))

def _dual_update(t,accept,mu,hbar,log_eps_bar,target=.8,gamma=.05,t0=10,kappa=.75):
    eta=1/(t+t0);hbar=(1-eta)*hbar+eta*(target-accept);logeps=mu-(math.sqrt(t)/gamma)*hbar;w=t**(-kappa);logbar=w*logeps+(1-w)*log_eps_bar
    return math.exp(logeps),hbar,logbar

def _diagnose(chains,warmup,names,payload):
    d={'chains':chains.tolist(),'warmup':warmup,'parameter_names':names,'max_rhat':float(payload.get('max_rhat',1.01)),'min_bulk_ess':float(payload.get('min_bulk_ess',400)),'min_tail_ess':float(payload.get('min_tail_ess',400))}
    return postdiag.diagnose_mcmc(d)

def hmc(payload:Mapping[str,Any])->dict[str,Any]:
    names,pm,ps,R,D,S,pred,logp_phys=_problem(payload);d=len(names);logp=lambda z: logp_phys(pm+ps*np.asarray(z,dtype=float));chains=int(payload.get('chains',4));warm=int(payload.get('warmup',800));draws=int(payload.get('draws',1200));L=int(payload.get('leapfrog_steps',12));seed=int(payload.get('seed',1729));target=float(payload.get('target_accept',.8))
    if chains<2 or chains>8 or warm<100 or draws<200 or L<1 or L>100 or chains*(warm+draws)*d>MAX_VALUES:raise InputError('invalid/oversized HMC settings')
    rng=np.random.Generator(np.random.PCG64(seed));arr=np.empty((chains,warm+draws,d));rates=[];divs=[];eps_out=[]
    for c in range(chains):
        q=rng.normal(scale=.25,size=d);eps=float(payload.get('step_size',_find_epsilon(q,logp,rng)));mu=math.log(10*eps);hbar=0.;logbar=math.log(eps);acc=0;dv=0;dv_warm=0
        for t in range(1,warm+draws+1):
            p0=rng.normal(size=d);q0=q.copy();lp0=logp(q0);p=p0.copy();qn=q0.copy();lpn=lp0
            for _ in range(L):qn,p,lpn=_leapfrog(qn,p,eps,logp)
            if math.isfinite(lpn):
                loga=lpn-.5*np.dot(p,p)-(lp0-.5*np.dot(p0,p0));a=min(1.0,math.exp(min(0.0,loga)))
            else:a=0.;loga=-math.inf
            if not math.isfinite(loga) or abs(loga)>1000:
                if t<=warm: dv_warm+=1
                else: dv+=1
            if rng.random()<a:q=qn;acc+=1
            if t<=warm:
                eps,hbar,logbar=_dual_update(t,a,mu,hbar,logbar,target)
                if t==warm:eps=math.exp(logbar)
            arr[c,t-1]=pm+ps*q
        rates.append(acc/(warm+draws));divs.append(dv);eps_out.append(eps);
        if 'warm_divs' not in locals(): warm_divs=[]
        warm_divs.append(dv_warm)
    diag=_diagnose(arr,warm,names,payload);post=arr[:,warm:].reshape(-1,d);status='POSTERIOR_QUALIFIED' if diag['status']=='QUALIFIED' and sum(divs)==0 else 'POSTERIOR_NOT_QUALIFIED'
    return {'engine_version':ENGINE_VERSION,'kind':'hamiltonian_monte_carlo','status':status,'parameter_names':names,'posterior_mean':post.mean(0).tolist(),'posterior_covariance':np.atleast_2d(np.cov(post,rowvar=False,ddof=1)).tolist(),'posterior_sd':post.std(0,ddof=1).tolist(),'acceptance_rates':rates,'post_warmup_divergences':divs,'warmup_divergences':warm_divs,'step_sizes':eps_out,'diagnostics':diag,'chain_shape':list(arr.shape),'claim_boundary':'Qualification assesses numerical sampling under the supplied posterior; it does not validate the scientific model.'}

def _stop(qm,qp,pm,pp):
    dq=qp-qm;return float(dq@pm)>=0 and float(dq@pp)>=0

def _build(q,p,logu,v,j,eps,logp,joint0,rng,delta_max=1000.):
    if j==0:
        q1,p1,lp1=_leapfrog(q,p,v*eps,logp);joint1=lp1-.5*np.dot(p1,p1) if math.isfinite(lp1) else -math.inf
        n=int(logu<=joint1);s=int(logu-delta_max<joint1);alpha=min(1.,math.exp(min(0.,joint1-joint0))) if math.isfinite(joint1) else 0.;div=int(joint1-joint0 < -delta_max or not math.isfinite(joint1))
        return q1,p1,q1,p1,q1,n,s,alpha,1,div
    qm,pm,qp,pp,qprop,n,s,a,na,div=_build(q,p,logu,v,j-1,eps,logp,joint0,rng,delta_max)
    if s:
        if v==-1:
            qm2,pm2,_,_,q2,n2,s2,a2,na2,d2=_build(qm,pm,logu,v,j-1,eps,logp,joint0,rng,delta_max);qm,pm=qm2,pm2
        else:
            _,_,qp2,pp2,q2,n2,s2,a2,na2,d2=_build(qp,pp,logu,v,j-1,eps,logp,joint0,rng,delta_max);qp,pp=qp2,pp2
        if n+n2>0 and rng.random()<n2/(n+n2):qprop=q2
        n+=n2;s=int(s2 and _stop(qm,qp,pm,pp));a+=a2;na+=na2;div+=d2
    return qm,pm,qp,pp,qprop,n,s,a,na,div

def nuts(payload:Mapping[str,Any])->dict[str,Any]:
    names,pm0,ps,R,D,S,pred,logp_phys=_problem(payload);d=len(names);logp=lambda z: logp_phys(pm0+ps*np.asarray(z,dtype=float));chains=int(payload.get('chains',4));warm=int(payload.get('warmup',800));draws=int(payload.get('draws',1200));depth=int(payload.get('max_treedepth',8));seed=int(payload.get('seed',1729));target=float(payload.get('target_accept',.8))
    if chains<2 or chains>8 or warm<100 or draws<200 or depth<2 or depth>12 or chains*(warm+draws)*d>MAX_VALUES:raise InputError('invalid/oversized NUTS settings')
    rng=np.random.Generator(np.random.PCG64(seed));arr=np.empty((chains,warm+draws,d));eps_out=[];divs=[];depth_hist=[];acc_hist=[]
    for c in range(chains):
        q=rng.normal(scale=.25,size=d);eps=float(payload.get('step_size',_find_epsilon(q,logp,rng)));mu=math.log(10*eps);hbar=0.;logbar=math.log(eps);dv=0;dv_warm=0;dh=[];ah=[]
        for t in range(1,warm+draws+1):
            p0=rng.normal(size=d);lp0=logp(q);joint0=lp0-.5*np.dot(p0,p0);logu=joint0-rng.exponential();qm=qp=q.copy();pm=pp=p0.copy();qprop=q.copy();n=1;s=1;j=0;alpha=0.;na=0;iterdiv=0
            while s and j<depth:
                v=-1 if rng.random()<.5 else 1
                if v==-1:qm,pm,_,_,qc,nc,sc,ac,nac,dc=_build(qm,pm,logu,v,j,eps,logp,joint0,rng)
                else:_,_,qp,pp,qc,nc,sc,ac,nac,dc=_build(qp,pp,logu,v,j,eps,logp,joint0,rng)
                if sc and n+nc>0 and rng.random()<nc/(n+nc):qprop=qc
                n+=nc;s=int(sc and _stop(qm,qp,pm,pp));alpha+=ac;na+=nac;iterdiv+=dc;j+=1
            q=qprop;ar=alpha/max(1,na);
            if iterdiv>0:
                if t<=warm: dv_warm+=1
                else: dv+=1
            dh.append(j);ah.append(ar)
            if t<=warm:
                eps,hbar,logbar=_dual_update(t,ar,mu,hbar,logbar,target)
                if t==warm:eps=math.exp(logbar)
            arr[c,t-1]=pm0+ps*q
        eps_out.append(eps);divs.append(dv);depth_hist.append(dh);acc_hist.append(float(np.mean(ah)));
        if 'nuts_warm_divs' not in locals(): nuts_warm_divs=[]
        nuts_warm_divs.append(dv_warm)
    diag=_diagnose(arr,warm,names,payload);post=arr[:,warm:].reshape(-1,d);status='POSTERIOR_QUALIFIED' if diag['status']=='QUALIFIED' and sum(divs)==0 else 'POSTERIOR_NOT_QUALIFIED'
    return {'engine_version':ENGINE_VERSION,'kind':'no_u_turn_sampler','status':status,'parameter_names':names,'posterior_mean':post.mean(0).tolist(),'posterior_covariance':np.atleast_2d(np.cov(post,rowvar=False,ddof=1)).tolist(),'posterior_sd':post.std(0,ddof=1).tolist(),'mean_acceptance_stat':acc_hist,'post_warmup_divergent_iterations':divs,'warmup_divergent_iterations':nuts_warm_divs,'final_step_sizes':eps_out,'max_treedepth':depth,'mean_tree_depth_by_chain':[float(np.mean(x)) for x in depth_hist],'diagnostics':diag,'chain_shape':list(arr.shape),'claim_boundary':'This is an internal NUTS implementation with dual-averaging adaptation for unconstrained normal-prior calibration models; diagnostics do not establish scientific model adequacy.'}
