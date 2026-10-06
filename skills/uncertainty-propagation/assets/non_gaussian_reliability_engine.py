#!/usr/bin/env python3
"""Non-Gaussian reliability methods for Uncertainty Lab v0.8.

Qualified routes:
- randomized scrambled-Sobol QMC for explicit Gaussian/t copulas and
  multivariate normal/t distributions;
- directional simulation in an independent latent Gaussian space for
  multivariate normal and Gaussian-copula (Nataf-style) models.

No covariance-only non-Gaussian joint law is inferred.
"""
from __future__ import annotations
import math
from typing import Any, Mapping
import numpy as np
from scipy import stats, optimize
from scipy.stats import qmc
import joint_distribution_engine as joint

ENGINE_VERSION='0.8.0'
MAX_DIM=32
MAX_QMC_POWER=21
MAX_DIRECTIONS=200_000
class InputError(ValueError): pass

def _expr(payload):
    s=str(payload.get('limit_state','')).strip()
    if not s: raise InputError('limit_state expression is required; failure is g(x)<=0')
    return joint._compile(s)

def _eval(node,names,X):
    env={n:X[:,i] for i,n in enumerate(names)}
    with np.errstate(all='ignore'): y=np.asarray(joint._eval(node,env),dtype=float)
    if y.ndim==0:y=np.full(X.shape[0],float(y))
    if y.shape!=(X.shape[0],) or not np.isfinite(y).all():raise InputError('limit_state produced invalid/non-finite values')
    return y

def _chol_pd(a,name):
    a=np.asarray(a,dtype=float)
    try:a=joint.validate_covariance(a,name)
    except Exception as exc:raise InputError(str(exc)) from exc
    try:return np.linalg.cholesky(a)
    except np.linalg.LinAlgError as exc:raise InputError(f'{name} must be positive definite for this transform') from exc

def _joint_setup(payload:Mapping[str,Any]):
    names=list(payload.get('variables') or []);d=len(names)
    if not 1<=d<=MAX_DIM or len(set(names))!=d:raise InputError(f'variables must contain 1..{MAX_DIM} unique names')
    jm=dict(payload.get('joint_model') or {});kind=str(jm.get('kind','')).lower()
    if kind=='gaussian_copula':
        marg=list(payload.get('marginals') or jm.get('marginals') or [])
        if len(marg)!=d:raise InputError('one marginal specification required per variable')
        R=joint.validate_correlation(jm.get('copula_correlation'))
        if R.shape!=(d,d):raise InputError('copula correlation dimension mismatch')
        L=_chol_pd(R,'copula correlation')
        return names,kind,d,marg,L,jm
    if kind=='t_copula':
        marg=list(payload.get('marginals') or jm.get('marginals') or [])
        if len(marg)!=d:raise InputError('one marginal specification required per variable')
        R=joint.validate_correlation(jm.get('copula_correlation'))
        if R.shape!=(d,d):raise InputError('copula correlation dimension mismatch')
        df=float(jm.get('df',0));
        if df<=0:raise InputError('t_copula requires df>0')
        L=_chol_pd(R,'copula correlation')
        return names,kind,d+1,marg,L,jm
    if kind=='multivariate_normal':
        mu=np.asarray(jm.get('mean'),dtype=float);C=np.asarray(jm.get('covariance'),dtype=float)
        if mu.shape!=(d,) or C.shape!=(d,d) or not np.isfinite(mu).all():raise InputError('multivariate_normal mean/covariance mismatch')
        L=_chol_pd(C,'multivariate_normal covariance')
        return names,kind,d,None,L,{**jm,'_mean':mu}
    if kind=='multivariate_t':
        mu=np.asarray(jm.get('mean'),dtype=float);S=np.asarray(jm.get('scale_matrix'),dtype=float);df=float(jm.get('df',0))
        if mu.shape!=(d,) or S.shape!=(d,d) or df<=0 or not np.isfinite(mu).all():raise InputError('invalid multivariate_t parameters')
        L=_chol_pd(S,'multivariate_t scale matrix')
        return names,kind,d+1,None,L,{**jm,'_mean':mu}
    raise InputError('qualified reliability joint_model.kind must be gaussian_copula, t_copula, multivariate_normal, or multivariate_t')

def _from_unit(U,setup):
    names,kind,qdim,marg,L,jm=setup;d=len(names);eps=np.finfo(float).eps;U=np.clip(np.asarray(U,dtype=float),eps,1-eps)
    if U.ndim!=2 or U.shape[1]!=qdim:raise InputError('unit-cube sample dimension mismatch')
    if kind=='gaussian_copula':
        z=stats.norm.ppf(U[:,:d])@L.T;uc=stats.norm.cdf(z)
        return np.column_stack([joint._marginal_ppf(uc[:,i],marg[i]) for i in range(d)])
    if kind=='t_copula':
        df=float(jm['df']);z=stats.norm.ppf(U[:,:d])@L.T;chi=stats.chi2.ppf(U[:,d],df);t=z/np.sqrt(chi[:,None]/df);uc=stats.t.cdf(t,df)
        return np.column_stack([joint._marginal_ppf(uc[:,i],marg[i]) for i in range(d)])
    if kind=='multivariate_normal':
        z=stats.norm.ppf(U[:,:d]);return np.asarray(jm['_mean'])+z@L.T
    if kind=='multivariate_t':
        df=float(jm['df']);z=stats.norm.ppf(U[:,:d])@L.T;chi=stats.chi2.ppf(U[:,d],df);return np.asarray(jm['_mean'])+z/np.sqrt(chi[:,None]/df)
    raise AssertionError(kind)

def randomized_qmc(payload:Mapping[str,Any])->dict[str,Any]:
    setup=_joint_setup(payload);names,kind,qdim,_,_,_=setup;node=_expr(payload)
    power=int(payload.get('power',15));reps=int(payload.get('replicates',8));seed=int(payload.get('seed',1729))
    if power<8 or power>MAX_QMC_POWER or reps<2 or reps>64:raise InputError('power must be in [8,21] and replicates in [2,64]')
    ps=[]
    for r in range(reps):
        eng=qmc.Sobol(d=qdim,scramble=True,seed=seed+104729*r);U=eng.random_base2(power);X=_from_unit(U,setup);g=_eval(node,names,X);ps.append(float(np.mean(g<=0)))
    a=np.asarray(ps);mean=float(a.mean());se=float(a.std(ddof=1)/math.sqrt(reps));n=2**power
    return {'engine_version':ENGINE_VERSION,'kind':'randomized_scrambled_sobol_reliability','status':'PASS','failure_probability':mean,
            'standard_error_across_scrambles':se,'replicate_probabilities':a.tolist(),'replicates':reps,'samples_per_replicate':n,'seed':seed,
            'joint_model_kind':kind,'failure_definition':'limit_state <= 0',
            'claim_boundary':'Standard error is across independent Owen-scrambled Sobol replicates; it is not an iid Bernoulli Monte Carlo SE.'}

def _latent_to_x(U,setup):
    names,kind,_,marg,L,jm=setup;U=np.asarray(U,dtype=float);d=len(names)
    if U.ndim==1:U=U.reshape(1,-1)
    if kind=='multivariate_normal':return np.asarray(jm['_mean'])+U@L.T
    if kind=='gaussian_copula':
        z=U@L.T;uc=stats.norm.cdf(z);return np.column_stack([joint._marginal_ppf(uc[:,i],marg[i]) for i in range(d)])
    raise InputError('directional simulation currently requires multivariate_normal or gaussian_copula')

def directional(payload:Mapping[str,Any])->dict[str,Any]:
    setup=_joint_setup(payload);names,kind,_,_,_,_=setup;d=len(names)
    if kind not in {'multivariate_normal','gaussian_copula'}:raise InputError('directional simulation is qualified only for Gaussian latent transforms')
    node=_expr(payload);nd=int(payload.get('directions',20000));reps=int(payload.get('replicates',4));seed=int(payload.get('seed',1729))
    if nd<500 or nd>MAX_DIRECTIONS or reps<2 or reps>20:raise InputError('directions must be in [500,200000], replicates in [2,20]')
    max_tail=float(payload.get('max_radial_tail',1e-12))
    if not 0<max_tail<1e-4:raise InputError('max_radial_tail must be in (0,1e-4)')
    rmax=float(stats.chi.isf(max_tail,df=d));origin=_latent_to_x(np.zeros(d),setup);g0=float(_eval(node,names,origin)[0])
    if not g0>0:raise InputError('directional simulation requires a safe latent origin g(0)>0')
    probs=[];cross_rates=[]
    for rep in range(reps):
        rng=np.random.Generator(np.random.PCG64(seed+rep*1000003));D=rng.normal(size=(nd,d));D/=np.linalg.norm(D,axis=1)[:,None];contrib=np.zeros(nd);cross=0
        # Coarse radial scan is vectorized across directions, then roots are refined individually.
        # First crossing is authoritative only under the declared star-shaped assumption.
        grid=np.linspace(0,rmax,25)
        G=np.empty((len(grid),nd),dtype=float)
        for k,rr in enumerate(grid):
            X=_latent_to_x(D*rr,setup);G[k]=_eval(node,names,X)
        for i,vec in enumerate(D):
            j=None
            for k in range(1,len(grid)):
                if G[k,i]<=0<G[k-1,i]:j=k;break
            if j is None:continue
            root=optimize.brentq(lambda rr: float(_eval(node,names,_latent_to_x(rr*vec,setup))[0]),grid[j-1],grid[j],xtol=1e-10,rtol=1e-10,maxiter=80)
            contrib[i]=float(stats.chi.sf(root,df=d));cross+=1
        probs.append(float(contrib.mean()));cross_rates.append(cross/nd)
    a=np.asarray(probs);return {'engine_version':ENGINE_VERSION,'kind':'latent_gaussian_directional_simulation','status':'PASS','failure_probability':float(a.mean()),
        'standard_error_across_direction_replicates':float(a.std(ddof=1)/math.sqrt(reps)),'replicate_probabilities':a.tolist(),'directions_per_replicate':nd,'replicates':reps,
        'crossing_fraction_by_replicate':cross_rates,'max_radius':rmax,'ignored_radial_tail_bound_per_direction':max_tail,'joint_model_kind':kind,'seed':seed,
        'assumptions':['safe latent origin','failure set is star-shaped along sampled rays / first outward crossing defines failure beyond the crossing'],
        'claim_boundary':'Directional simulation is not qualified when rays can leave and re-enter the failure domain without an explicit multi-crossing treatment.'}
