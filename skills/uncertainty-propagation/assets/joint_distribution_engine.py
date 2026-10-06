#!/usr/bin/env python3
"""Explicit joint-distribution propagation for correlated non-Gaussian inputs.

Supported dependence routes are deliberately explicit: empirical/posterior joint samples,
Gaussian copula, t copula, multivariate normal and multivariate t.  A covariance or Pearson
correlation matrix by itself never defines a non-Gaussian joint law.
"""
from __future__ import annotations
import ast, math
from typing import Any, Mapping, Sequence
import numpy as np
from scipy import stats

ENGINE_VERSION="0.5.0"
MAX_SAMPLES=2_000_000
MAX_VARIABLES=256
MAX_EXPRESSIONS=32
class InputError(ValueError): pass

_NPF={"sqrt":np.sqrt,"exp":np.exp,"log":np.log,"sin":np.sin,"cos":np.cos,"tan":np.tan,
      "asin":np.arcsin,"acos":np.arccos,"atan":np.arctan,"sinh":np.sinh,"cosh":np.cosh,"tanh":np.tanh,"abs":np.abs}

def _eval(node,env):
    if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)): return node.value
    if isinstance(node,ast.Name):
        if node.id in env:return env[node.id]
        if node.id=='pi':return math.pi
        if node.id=='e':return math.e
        raise InputError(f"unknown name {node.id!r}")
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
        v=_eval(node.operand,env); return v if isinstance(node.op,ast.UAdd) else -v
    if isinstance(node,ast.BinOp):
        a=_eval(node.left,env);b=_eval(node.right,env)
        if isinstance(node.op,ast.Add):return a+b
        if isinstance(node.op,ast.Sub):return a-b
        if isinstance(node.op,ast.Mult):return a*b
        if isinstance(node.op,ast.Div):return a/b
        if isinstance(node.op,ast.Pow):return a**b
        raise InputError('operator not allowed')
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords and len(node.args)==1:
        fn=_NPF.get(node.func.id)
        if fn is None: raise InputError(f"function not allowed: {node.func.id}")
        return fn(_eval(node.args[0],env))
    raise InputError(f"syntax not allowed: {type(node).__name__}")

def _compile(s):
    if not isinstance(s,str) or not s.strip() or len(s)>5000: raise InputError('invalid expression')
    try:return ast.parse(s,mode='eval').body
    except SyntaxError as e: raise InputError(f'invalid expression syntax: {e}') from e

def validate_correlation(corr:Any)->np.ndarray:
    r=np.asarray(corr,dtype=float)
    if r.ndim!=2 or r.shape[0]!=r.shape[1] or not np.isfinite(r).all(): raise InputError('correlation must be finite square')
    if np.max(np.abs(r-r.T))>1e-10: raise InputError('correlation is not symmetric')
    if np.max(np.abs(np.diag(r)-1.0))>1e-10: raise InputError('correlation diagonal must equal 1')
    if np.max(np.abs(r))>1+1e-10: raise InputError('correlation entries outside [-1,1]')
    ev=np.linalg.eigvalsh((r+r.T)/2)
    tol=1e-12+1e-10*max(1.0,float(np.max(np.abs(ev))))
    if ev[0] < -tol: raise InputError(f'correlation not PSD: min eigenvalue={ev[0]:g}')
    return (r+r.T)/2

def validate_covariance(cov:Any,name:str="covariance")->np.ndarray:
    c=np.asarray(cov,dtype=float)
    if c.ndim!=2 or c.shape[0]!=c.shape[1] or not np.isfinite(c).all(): raise InputError(f'{name} must be finite square')
    d=np.diag(c)
    if np.any(d<0): raise InputError(f'{name} has negative variance')
    pos=d>0
    if np.any(~pos) and (np.any(c[~pos,:]!=0) or np.any(c[:,~pos]!=0)): raise InputError(f'{name}: zero-variance row has nonzero covariance')
    if np.any(pos):
        sd=np.sqrt(d[pos]); raw=c[np.ix_(pos,pos)]/np.outer(sd,sd); scale=max(1.0,float(np.max(np.abs(raw))))
        if np.max(np.abs(raw-raw.T))>1e-12+1e-10*scale: raise InputError(f'{name} not symmetric on variance-normalized basis')
        r=(raw+raw.T)/2; ev=np.linalg.eigvalsh(r); tol=1e-12+1e-10*max(1.0,float(np.max(np.abs(ev))))
        if ev[0]<-tol: raise InputError(f'{name} not PSD on variance-normalized basis')
    return (c+c.T)/2

def _factor_psd(r:np.ndarray)->np.ndarray:
    ev,Q=np.linalg.eigh(r)
    scale=float(np.max(np.abs(ev))) if ev.size else 0.0
    tol=(1e-12+1e-10)*scale if scale>0 else 0.0
    if ev.size and ev[0]<-tol: raise InputError('matrix not PSD')
    keep=ev>tol
    return Q[:,keep] * np.sqrt(ev[keep])

def _marginal_ppf(u:np.ndarray,spec:Mapping[str,Any])->np.ndarray:
    kind=str(spec.get('kind','normal')).lower()
    eps=np.finfo(float).eps
    q=np.clip(np.asarray(u,dtype=float),eps,1-eps)
    if kind in {'normal','gaussian'}:
        loc=float(spec.get('mean',spec.get('loc',0.0)));scale=float(spec.get('std',spec.get('scale',1.0)))
        if not math.isfinite(loc) or not math.isfinite(scale) or scale<=0: raise InputError('normal marginal requires finite mean and positive standard deviation')
        return stats.norm.ppf(q,loc=loc,scale=scale)
    if kind in {'uniform','rectangular'}:
        if 'lower' in spec and 'upper' in spec: lo,hi=float(spec['lower']),float(spec['upper'])
        elif 'half_width' in spec:
            c=float(spec.get('value',spec.get('center',0.0)));h=float(spec['half_width']);lo,hi=c-h,c+h
        else: raise InputError('uniform requires lower/upper or half_width')
        if not lo<hi: raise InputError('uniform lower must be < upper')
        return lo+(hi-lo)*q
    if kind=='triangular':
        if {'lower','mode','upper'}<=set(spec): lo,mode,hi=map(float,(spec['lower'],spec['mode'],spec['upper']))
        elif 'half_width' in spec:
            c=float(spec.get('value',spec.get('center',0.0)));h=float(spec['half_width']);lo,mode,hi=c-h,c,c+h
        else: raise InputError('triangular requires lower/mode/upper or half_width')
        if not lo<=mode<=hi or lo==hi: raise InputError('invalid triangular parameters')
        return stats.triang.ppf(q,c=(mode-lo)/(hi-lo),loc=lo,scale=hi-lo)
    if kind in {'lognormal','log_normal'}:
        mu=float(spec.get('log_mean',spec.get('meanlog',0.0))); sig=float(spec.get('log_std',spec.get('sdlog',spec.get('sigma',1.0))))
        if sig<=0: raise InputError('lognormal log_std must be positive')
        return stats.lognorm.ppf(q,s=sig,scale=math.exp(mu))
    if kind=='beta':
        a=float(spec.get('a'));b=float(spec.get('b'));lo=float(spec.get('lower',0.0));hi=float(spec.get('upper',1.0))
        if a<=0 or b<=0 or not lo<hi: raise InputError('invalid beta parameters')
        return lo+(hi-lo)*stats.beta.ppf(q,a,b)
    if kind=='gamma':
        shape=float(spec.get('shape',spec.get('a'))); scale=float(spec.get('scale',1.0));loc=float(spec.get('loc',0.0))
        if shape<=0 or scale<=0: raise InputError('invalid gamma parameters')
        return stats.gamma.ppf(q,a=shape,loc=loc,scale=scale)
    if kind in {'student_t','t'}:
        df=float(spec.get('df'));loc=float(spec.get('loc',0.0));scale=float(spec.get('scale',1.0))
        if df<=0 or scale<=0: raise InputError('invalid t marginal parameters')
        return stats.t.ppf(q,df=df,loc=loc,scale=scale)
    if kind=='empirical':
        vals=np.sort(np.asarray(spec.get('values'),dtype=float))
        if vals.ndim!=1 or vals.size<2 or not np.isfinite(vals).all(): raise InputError('empirical marginal requires >=2 finite values')
        pos=q*(vals.size-1);i=np.floor(pos).astype(int);j=np.minimum(i+1,vals.size-1);w=pos-i
        return vals[i]*(1-w)+vals[j]*w
    raise InputError(f'unsupported marginal kind {kind!r}')

def _sample_copula(kind:str,corr:np.ndarray,n:int,rng:np.random.Generator,df:float|None=None)->np.ndarray:
    L=_factor_psd(corr); z=rng.normal(size=(n,L.shape[1]))@L.T
    if kind=='gaussian_copula': return stats.norm.cdf(z)
    if kind=='t_copula':
        if df is None or df<=0: raise InputError('t_copula requires df>0')
        chi=rng.chisquare(df,size=n); t=z/np.sqrt(chi[:,None]/df); return stats.t.cdf(t,df=df)
    raise InputError('unsupported copula')

def generate_inputs(payload:Mapping[str,Any])->tuple[list[str],np.ndarray,dict[str,Any]]:
    names=list(payload.get('variables') or [])
    joint=dict(payload.get('joint_model') or {})
    kind=str(joint.get('kind','')).lower()
    if kind in {'empirical_samples','posterior_samples','joint_samples'}:
        X=np.asarray(payload.get('samples',joint.get('samples')),dtype=float)
        if X.ndim!=2 or X.shape[0]<2 or X.shape[0]>MAX_SAMPLES or X.shape[1]>MAX_VARIABLES or not np.isfinite(X).all(): raise InputError('joint samples must be finite bounded N x D')
        if not names: names=[f'x{i}' for i in range(X.shape[1])]
        if X.shape[1]!=len(names) or len(set(names))!=len(names): raise InputError('sample columns/variables mismatch')
        return names,X,{"kind":kind,"n":int(X.shape[0]),"sampling":"provided_joint_samples"}
    if not names or len(set(names))!=len(names) or len(names)>MAX_VARIABLES: raise InputError('variables must be unique and bounded')
    n=int(payload.get('n',joint.get('n',100000)));seed=int(payload.get('seed',joint.get('seed',1729)))
    if n<100 or n>MAX_SAMPLES: raise InputError(f'n must be in [100,{MAX_SAMPLES}]')
    rng=np.random.Generator(np.random.PCG64(seed));d=len(names)
    if kind in {'gaussian_copula','t_copula'}:
        marg=list(payload.get('marginals') or joint.get('marginals') or [])
        if len(marg)!=d: raise InputError('one marginal specification required per variable')
        corr=validate_correlation(joint.get('copula_correlation'))
        if corr.shape!=(d,d): raise InputError('copula correlation dimension mismatch')
        u=_sample_copula(kind,corr,n,rng,float(joint['df']) if kind=='t_copula' else None)
        X=np.column_stack([_marginal_ppf(u[:,i],marg[i]) for i in range(d)])
        return names,X,{"kind":kind,"seed":seed,"n":n,"copula_correlation":corr.tolist(),"correlation_semantics":"latent copula correlation, not asserted Pearson correlation of transformed marginals",**({"df":float(joint['df'])} if kind=='t_copula' else {})}
    if kind=='multivariate_normal':
        mean=np.asarray(joint.get('mean'),dtype=float);cov=np.asarray(joint.get('covariance'),dtype=float)
        if mean.shape!=(d,) or cov.shape!=(d,d):raise InputError('mean/covariance dimension mismatch')
        cov=validate_covariance(cov,'multivariate_normal covariance')
        X=rng.multivariate_normal(mean,cov,size=n,check_valid='raise')
        return names,X,{"kind":kind,"seed":seed,"n":n}
    if kind=='multivariate_t':
        mean=np.asarray(joint.get('mean'),dtype=float);scale=np.asarray(joint.get('scale_matrix'),dtype=float);df=float(joint.get('df'))
        if mean.shape!=(d,) or scale.shape!=(d,d) or df<=0: raise InputError('invalid multivariate_t parameters')
        R=validate_covariance(scale,'multivariate_t scale matrix')
        L=_factor_psd(R);z=rng.normal(size=(n,L.shape[1]))@L.T;chi=rng.chisquare(df,size=n)
        X=mean+z/np.sqrt(chi[:,None]/df)
        return names,X,{"kind":kind,"seed":seed,"n":n,"df":df}
    raise InputError('joint_model.kind must explicitly specify empirical_samples, gaussian_copula, t_copula, multivariate_normal, or multivariate_t')

def propagate(payload:Mapping[str,Any])->dict[str,Any]:
    names,X,jmeta=generate_inputs(payload)
    expressions=list(payload.get('expressions') or ([payload['expression']] if 'expression' in payload else []))
    if not expressions or len(expressions)>MAX_EXPRESSIONS: raise InputError(f'expression count must be in [1,{MAX_EXPRESSIONS}]')
    compiled=[_compile(e) for e in expressions]
    env={n:X[:,i] for i,n in enumerate(names)}; outs=[]; valid=np.ones(X.shape[0],dtype=bool)
    for c in compiled:
        with np.errstate(all='ignore'):
            y=np.asarray(_eval(c,env),dtype=float)
        if y.ndim==0:y=np.full(X.shape[0],float(y))
        if y.shape!=(X.shape[0],):raise InputError('expression did not produce one value per sample')
        valid &= np.isfinite(y);outs.append(y)
    invalid=int(np.sum(~valid));n=X.shape[0]
    out={"engine_version":ENGINE_VERSION,"method":"explicit_joint_distribution_propagation","variable_order":names,"joint_model":jmeta,
         "n":int(n),"invalid_count":invalid,"invalid_fraction":invalid/n,"status":"ok" if invalid==0 else ("all_samples_invalid" if invalid==n else "domain_violations")}
    if invalid==n:return out
    Y=np.vstack([y[valid] for y in outs]);V=X[valid]
    cov=np.atleast_2d(np.cov(Y,ddof=1));mean=Y.mean(axis=1);sd=Y.std(axis=1,ddof=1)
    out.update({"mean":mean.tolist(),"std":sd.tolist(),"output_covariance":cov.tolist(),"input_sample_covariance":np.atleast_2d(np.cov(V,rowvar=False,ddof=1)).tolist()})
    den=np.outer(sd,sd)
    with np.errstate(invalid='ignore',divide='ignore'): corr=np.divide(cov,den,out=np.full_like(cov,np.nan),where=den>0)
    out['output_correlation']=[[None if not np.isfinite(v) else float(v) for v in row] for row in corr]
    p=float(payload.get('coverage_probability',payload.get('coverage',{}).get('probability',0.95) if isinstance(payload.get('coverage'),Mapping) else 0.95))
    if not 0<p<1: raise InputError('coverage probability must be in (0,1)')
    lo=(1-p)/2;hi=1-lo
    out['equal_tailed_coverage_interval']={"probability":p,"lower":np.quantile(Y,lo,axis=1).tolist(),"upper":np.quantile(Y,hi,axis=1).tolist(),"conditional_on_valid":bool(invalid)}
    if invalid:out['conditional_on_valid']=True
    return out
