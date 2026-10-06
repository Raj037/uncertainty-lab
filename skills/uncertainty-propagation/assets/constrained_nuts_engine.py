#!/usr/bin/env python3
"""Constrained transformed-space NUTS for Uncertainty Lab v0.9.

Qualified parameter families are expressed through explicit smooth transforms from
unconstrained standard-normal latent variables. This avoids clipping/reflection at
boundaries and makes the induced prior measure explicit.

Supported parameter specs:
- normal: unconstrained real, physical = mean + std*z
- lognormal: positive, physical = exp(log_mean + log_std*z)
- logit_normal: bounded (lower, upper), logistic transform of normal latent
- simplex_logistic_normal: K simplex components from K-1 normal latents

The sampler uses a diagonal Euclidean metric estimated from warm-up latent draws.
The metric is frozen before retained draws. Numerical posterior diagnostics are
provided by the qualified v0.6 posterior_diagnostics engine.
"""
from __future__ import annotations
import math
from typing import Any, Mapping
import numpy as np
from scipy import linalg
import calibration_engine as cal
import joint_distribution_engine as joint
import posterior_diagnostics as postdiag

ENGINE_VERSION = "0.9.0"
MAX_PHYSICAL_PARAMETERS = 24
MAX_LATENT_DIM = 20
MAX_VALUES = 3_000_000
class InputError(ValueError): pass


def _sigmoid(x):
    if x >= 0:
        e=math.exp(-x); return 1/(1+e)
    e=math.exp(x); return e/(1+e)


def _compile_parameterization(payload: Mapping[str,Any]):
    specs=list(payload.get("parameter_specs") or [])
    if not specs: raise InputError("parameter_specs is required")
    physical_names=[]; entries=[]; latent_dim=0
    for raw in specs:
        s=dict(raw); kind=str(s.get("kind","normal")).lower()
        if kind in {"normal","real_normal"}:
            name=str(s.get("name","")).strip(); mean=float(s.get("mean",0)); std=float(s.get("std",1))
            if not name or name in physical_names or not math.isfinite(mean) or not math.isfinite(std) or std<=0: raise InputError("invalid normal parameter spec")
            physical_names.append(name); entries.append((kind,[name],latent_dim,{"mean":mean,"std":std})); latent_dim+=1
        elif kind in {"lognormal","positive_lognormal"}:
            name=str(s.get("name","")).strip(); mu=float(s.get("log_mean",s.get("meanlog",0))); sd=float(s.get("log_std",s.get("sdlog",1)))
            if not name or name in physical_names or not math.isfinite(mu) or not math.isfinite(sd) or sd<=0: raise InputError("invalid lognormal parameter spec")
            physical_names.append(name); entries.append((kind,[name],latent_dim,{"mu":mu,"sd":sd})); latent_dim+=1
        elif kind in {"logit_normal","bounded_logit_normal"}:
            name=str(s.get("name","")).strip(); lo=float(s.get("lower")); hi=float(s.get("upper")); mu=float(s.get("logit_mean",0)); sd=float(s.get("logit_std",1))
            if not name or name in physical_names or not lo<hi or not math.isfinite(mu) or not math.isfinite(sd) or sd<=0: raise InputError("invalid bounded logit-normal parameter spec")
            physical_names.append(name); entries.append((kind,[name],latent_dim,{"lo":lo,"hi":hi,"mu":mu,"sd":sd})); latent_dim+=1
        elif kind in {"simplex_logistic_normal","simplex"}:
            names=[str(x).strip() for x in (s.get("names") or [])]; k=len(names)
            if k<2 or len(set(names))!=k or any((not n or n in physical_names) for n in names): raise InputError("simplex requires >=2 unique new names")
            loc=np.asarray(s.get("latent_mean",[0.0]*(k-1)),dtype=float); scale=np.asarray(s.get("latent_std",[1.0]*(k-1)),dtype=float)
            if loc.shape!=(k-1,) or scale.shape!=(k-1,) or not np.isfinite(loc).all() or not np.isfinite(scale).all() or np.any(scale<=0): raise InputError("invalid simplex latent parameters")
            physical_names.extend(names); entries.append((kind,names,latent_dim,{"loc":loc,"scale":scale})); latent_dim+=k-1
        else: raise InputError(f"unsupported parameter kind {kind!r}")
    if len(physical_names)>MAX_PHYSICAL_PARAMETERS or latent_dim>MAX_LATENT_DIM: raise InputError("parameter dimension exceeds qualified v0.9 limits")

    def transform(z):
        z=np.asarray(z,dtype=float)
        if z.shape!=(latent_dim,) or not np.isfinite(z).all(): raise InputError("latent vector mismatch")
        env={}; cursor=0
        for kind,names,start,meta in entries:
            if kind in {"normal","real_normal"}:
                env[names[0]]=meta["mean"]+meta["std"]*z[start]
            elif kind in {"lognormal","positive_lognormal"}:
                t=meta["mu"]+meta["sd"]*z[start]
                if t>700: raise InputError("lognormal transform overflow")
                env[names[0]]=math.exp(t)
            elif kind in {"logit_normal","bounded_logit_normal"}:
                t=meta["mu"]+meta["sd"]*z[start]; u=_sigmoid(t); env[names[0]]=meta["lo"]+(meta["hi"]-meta["lo"])*u
            else:
                k=len(names); a=meta["loc"]+meta["scale"]*z[start:start+k-1]
                m=max(0.0,float(np.max(a))); ea=np.exp(a-m); elast=math.exp(-m); den=float(np.sum(ea)+elast); vals=np.r_[ea/den,elast/den]
                for n,v in zip(names,vals): env[n]=float(v)
        return env
    return physical_names, latent_dim, transform


def _problem(payload):
    names,d,transform=_compile_parameterization(payload)
    y=np.asarray(payload.get("observed"),dtype=float)
    if y.ndim!=1 or y.size<2 or not np.isfinite(y).all(): raise InputError("observed must be finite vector")
    n=y.size; R,D,S=cal._obs_cov(payload,n); cf=linalg.cho_factor(S,lower=True); logdet=2*float(np.sum(np.log(np.diag(cf[0]))))
    data=dict(payload.get("data") or {})
    for k,v in list(data.items()):
        a=np.asarray(v,dtype=float)
        if a.shape!=(n,) or not np.isfinite(a).all(): raise InputError(f"data[{k}] must be finite length n")
        data[k]=a
    expr=str(payload.get("model_expression","")).strip()
    if not expr: raise InputError("model_expression required")
    node=joint._compile(expr)
    def pred_z(z):
        phys=transform(z); env={**data,**phys}
        with np.errstate(all="ignore"): p=np.asarray(joint._eval(node,env),dtype=float)
        if p.ndim==0: p=np.full(n,float(p))
        if p.shape!=(n,) or not np.isfinite(p).all(): raise InputError("model expression produced invalid predictions")
        return p,phys
    def logp(z):
        z=np.asarray(z,dtype=float)
        if z.shape!=(d,) or not np.isfinite(z).all(): return -math.inf
        # z is the explicit prior latent variable: N(0,I). This is equivalent to
        # including the transform Jacobian in physical-space density.
        lp=float(-.5*np.dot(z,z)-.5*d*math.log(2*math.pi))
        try: p,_=pred_z(z)
        except Exception: return -math.inf
        r=y-p; quad=float(r@linalg.cho_solve(cf,r,check_finite=False))
        return lp-.5*(n*math.log(2*math.pi)+logdet+quad)
    return names,d,transform,pred_z,logp


def _grad(logp,q):
    q=np.asarray(q,dtype=float); g=np.empty_like(q); eps=np.finfo(float).eps**.2
    for i,x in enumerate(q):
        h=eps*max(1.0,abs(float(x))); h=max(h,32*abs(float(np.spacing(float(x)))),np.finfo(float).tiny)
        e=np.zeros_like(q); e[i]=h
        fs=[logp(q+2*e),logp(q+e),logp(q-e),logp(q-2*e)]
        if not all(map(math.isfinite,fs)): raise InputError("non-finite posterior during gradient")
        g[i]=(-fs[0]+8*fs[1]-8*fs[2]+fs[3])/(12*h)
    return g


def _kinetic(p,inv_mass): return .5*float(np.dot(p*inv_mass,p))
def _momentum(rng,inv_mass): return rng.normal(size=len(inv_mass))/np.sqrt(inv_mass)

def _leapfrog(q,p,eps,logp,inv_mass):
    q=q.copy(); p=p.copy(); p+=.5*eps*_grad(logp,q); q+=eps*(inv_mass*p); lp=logp(q)
    if not math.isfinite(lp): return q,p,-math.inf
    p+=.5*eps*_grad(logp,q); return q,p,lp


def _find_epsilon(q,logp,rng,inv_mass):
    p=_momentum(rng,inv_mass); lp=logp(q); q1,p1,lp1=_leapfrog(q,p,1.0,logp,inv_mass)
    if not math.isfinite(lp1): return .25
    loga=lp1-_kinetic(p1,inv_mass)-(lp-_kinetic(p,inv_mass)); direction=1 if loga>math.log(.5) else -1; eps=1.0
    for _ in range(18):
        cond=(loga>math.log(.5)) if direction==1 else (loga<math.log(.5))
        if not cond: break
        eps*=2.0**direction
        if not 1e-6<=eps<=5: break
        q1,p1,lp1=_leapfrog(q,p,eps,logp,inv_mass)
        loga=-math.inf if not math.isfinite(lp1) else lp1-_kinetic(p1,inv_mass)-(lp-_kinetic(p,inv_mass))
    return min(2.0,max(1e-5,eps))


def _stop(qm,qp,pm,pp,inv_mass):
    dq=qp-qm; return float(dq@(inv_mass*pm))>=0 and float(dq@(inv_mass*pp))>=0


def _build(q,p,logu,v,j,eps,logp,joint0,rng,inv_mass,delta_max=1000.):
    if j==0:
        q1,p1,lp1=_leapfrog(q,p,v*eps,logp,inv_mass); joint1=lp1-_kinetic(p1,inv_mass) if math.isfinite(lp1) else -math.inf
        n=int(logu<=joint1); s=int(logu-delta_max<joint1); alpha=min(1.,math.exp(min(0.,joint1-joint0))) if math.isfinite(joint1) else 0.; div=int(joint1-joint0 < -delta_max or not math.isfinite(joint1))
        return q1,p1,q1,p1,q1,n,s,alpha,1,div
    qm,pm,qp,pp,qprop,n,s,a,na,div=_build(q,p,logu,v,j-1,eps,logp,joint0,rng,inv_mass,delta_max)
    if s:
        if v==-1:
            qm2,pm2,_,_,q2,n2,s2,a2,na2,d2=_build(qm,pm,logu,v,j-1,eps,logp,joint0,rng,inv_mass,delta_max); qm,pm=qm2,pm2
        else:
            _,_,qp2,pp2,q2,n2,s2,a2,na2,d2=_build(qp,pp,logu,v,j-1,eps,logp,joint0,rng,inv_mass,delta_max); qp,pp=qp2,pp2
        if n+n2>0 and rng.random()<n2/(n+n2): qprop=q2
        n+=n2; s=int(s2 and _stop(qm,qp,pm,pp,inv_mass)); a+=a2; na+=na2; div+=d2
    return qm,pm,qp,pp,qprop,n,s,a,na,div


def _dual_update(t,accept,mu,hbar,log_eps_bar,target=.8,gamma=.05,t0=10,kappa=.75):
    eta=1/(t+t0); hbar=(1-eta)*hbar+eta*(target-accept); logeps=mu-(math.sqrt(t)/gamma)*hbar; w=t**(-kappa); logbar=w*logeps+(1-w)*log_eps_bar
    return math.exp(logeps),hbar,logbar


def nuts(payload:Mapping[str,Any]):
    names,d,transform,pred,logp=_problem(payload)
    chains=int(payload.get("chains",4)); warm=int(payload.get("warmup",900)); draws=int(payload.get("draws",1000)); depth=int(payload.get("max_treedepth",8)); seed=int(payload.get("seed",1729)); target=float(payload.get("target_accept",.8))
    if not 2<=chains<=8 or warm<300 or draws<300 or not 2<=depth<=12 or chains*(warm+draws)*d>MAX_VALUES: raise InputError("invalid/oversized NUTS settings")
    rng=np.random.Generator(np.random.PCG64(seed)); latent=np.empty((chains,warm+draws,d)); divs=[]; warm_divs=[]; rates=[]; epss=[]; metrics=[]
    for c in range(chains):
        q=rng.normal(scale=.2,size=d); inv_mass=np.ones(d); eps=_find_epsilon(q,logp,rng,inv_mass); mu=math.log(10*eps); hbar=0.; logbar=math.log(eps); dv=dvw=0; acc=[]; warm_samples=[]
        for t in range(1,warm+draws+1):
            p0=_momentum(rng,inv_mass); lp0=logp(q); joint0=lp0-_kinetic(p0,inv_mass); logu=joint0-rng.exponential(); qm=qp=q.copy(); pm=pp=p0.copy(); qprop=q.copy(); n=1; active=1; j=0; alpha=0.; na=0; iterdiv=0
            while active and j<depth:
                v=-1 if rng.random()<.5 else 1
                if v==-1: qm,pm,_,_,qc,nc,sc,ac,nac,dc=_build(qm,pm,logu,v,j,eps,logp,joint0,rng,inv_mass)
                else: _,_,qp,pp,qc,nc,sc,ac,nac,dc=_build(qp,pp,logu,v,j,eps,logp,joint0,rng,inv_mass)
                if sc and n+nc>0 and rng.random()<nc/(n+nc): qprop=qc
                n+=nc; active=int(sc and _stop(qm,qp,pm,pp,inv_mass)); alpha+=ac; na+=nac; iterdiv+=dc; j+=1
            q=qprop; ar=alpha/max(1,na); acc.append(ar)
            if iterdiv:
                if t<=warm: dvw+=1
                else: dv+=1
            if t<=warm:
                warm_samples.append(q.copy()); eps,hbar,logbar=_dual_update(t,ar,mu,hbar,logbar,target)
                # Windowed diagonal metric adaptation. The inverse mass used by
                # dynamics is a clipped estimate of posterior latent covariance.
                if t>=200 and t%100==0 and t<warm:
                    W=np.asarray(warm_samples[max(0,len(warm_samples)-300):]); var=np.var(W,axis=0,ddof=1); inv_mass=np.clip(var,1e-3,1e3)
                    eps=_find_epsilon(q,logp,rng,inv_mass); mu=math.log(10*eps); hbar=0.; logbar=math.log(eps)
                if t==warm: eps=math.exp(logbar)
            latent[c,t-1]=q
        divs.append(dv); warm_divs.append(dvw); rates.append(float(np.mean(acc))); epss.append(float(eps)); metrics.append(inv_mass.tolist())
    phys=np.empty((chains,warm+draws,len(names)))
    for c in range(chains):
        for t in range(warm+draws):
            env=transform(latent[c,t]); phys[c,t]=[env[n] for n in names]
    diag=postdiag.diagnose_mcmc({"chains":phys.tolist(),"warmup":warm,"parameter_names":names,"max_rhat":float(payload.get("max_rhat",1.01)),"min_bulk_ess":float(payload.get("min_bulk_ess",300)),"min_tail_ess":float(payload.get("min_tail_ess",300))})
    post=phys[:,warm:].reshape(-1,len(names)); status="POSTERIOR_QUALIFIED" if diag["status"]=="QUALIFIED" and sum(divs)==0 else "POSTERIOR_NOT_QUALIFIED"
    return {"engine_version":ENGINE_VERSION,"kind":"transformed_nuts_diagonal_metric","status":status,"parameter_names":names,"posterior_mean":post.mean(0).tolist(),"posterior_sd":post.std(0,ddof=1).tolist(),"posterior_covariance":np.atleast_2d(np.cov(post,rowvar=False,ddof=1)).tolist(),"post_warmup_divergences":divs,"warmup_divergences":warm_divs,"mean_acceptance_stat":rates,"final_step_sizes":epss,"inverse_mass_diagonal":metrics,"diagnostics":diag,"chain_shape":list(phys.shape),"claim_boundary":"Constraints are represented by explicit generative latent transforms. Qualification concerns numerical sampling under the supplied model; it does not establish prior/model scientific adequacy."}
