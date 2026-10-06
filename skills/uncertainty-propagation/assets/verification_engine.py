#!/usr/bin/env python3
"""Three-engine derivative verification: symbolic, forward AD, finite difference."""
from __future__ import annotations
import ast,math
from typing import Any,Mapping,Sequence
import numpy as np
import sympy as sp
import autodiff_engine as ad
ENGINE_VERSION='0.5.0'
class InputError(ValueError):pass

_SPF={'sqrt':sp.sqrt,'exp':sp.exp,'log':sp.log,'sin':sp.sin,'cos':sp.cos,'tan':sp.tan,'asin':sp.asin,'acos':sp.acos,'atan':sp.atan,'sinh':sp.sinh,'cosh':sp.cosh,'tanh':sp.tanh,'abs':sp.Abs}
def _sym(node,env):
    if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)):return sp.Float(node.value) if isinstance(node.value,float) else sp.Integer(node.value)
    if isinstance(node,ast.Name):
        if node.id in env:return env[node.id]
        if node.id=='pi':return sp.pi
        if node.id=='e':return sp.E
        raise InputError(f'unknown name {node.id!r}')
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
        v=_sym(node.operand,env);return v if isinstance(node.op,ast.UAdd) else -v
    if isinstance(node,ast.BinOp):
        a=_sym(node.left,env);b=_sym(node.right,env)
        if isinstance(node.op,ast.Add):return a+b
        if isinstance(node.op,ast.Sub):return a-b
        if isinstance(node.op,ast.Mult):return a*b
        if isinstance(node.op,ast.Div):return a/b
        if isinstance(node.op,ast.Pow):return a**b
        raise InputError('operator not allowed')
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords and len(node.args)==1:
        fn=_SPF.get(node.func.id)
        if fn is None:raise InputError(f'function not allowed: {node.func.id}')
        return fn(_sym(node.args[0],env))
    raise InputError(f'syntax not allowed: {type(node).__name__}')

def _symbolic(expressions,names,mu):
    syms=[sp.Symbol(n,real=True) for n in names];env=dict(zip(names,syms));expr=[]
    for e in expressions:expr.append(_sym(ad.compile_expression(e),env))
    J=sp.Matrix(expr).jacobian(syms);Hs=[sp.hessian(e,syms) for e in expr]
    subs=dict(zip(syms,map(float,mu)))
    vals=np.array([float(e.evalf(subs=subs)) for e in expr]);j=np.array(J.evalf(subs=subs),dtype=float).reshape(len(expr),len(names));h=np.array([np.array(H.evalf(subs=subs),dtype=float) for H in Hs])
    if not np.isfinite(vals).all() or not np.isfinite(j).all() or not np.isfinite(h).all():raise InputError('symbolic derivative non-finite')
    return vals,j,h

def _finite_difference(expressions,names,mu,cov):
    m=len(expressions);n=len(names);J=np.zeros((m,n));steps=[];attempts=[];rel=float(np.finfo(float).eps**0.2)
    y=np.array([ad.evaluate_float(e,names,mu) for e in expressions])
    for k in range(n):
        sigma=math.sqrt(max(0,float(cov[k,k])));ch=max(abs(float(mu[k])),sigma)
        if ch==0:ch=1.0
        h=max(rel*ch,32*abs(float(np.spacing(float(mu[k])))),np.finfo(float).tiny);ok=False;err=None
        for a in range(13):
            try:
                pts=[]
                for mult in (2,1,-1,-2):
                    z=mu.copy();z[k]+=mult*h;pts.append(np.array([ad.evaluate_float(e,names,z) for e in expressions]))
                fp2,fp1,fm1,fm2=pts;ok=True;break
            except Exception as exc:
                err=exc;h*=.25
                if h<=32*abs(float(np.spacing(float(mu[k])))) or h<=np.finfo(float).tiny:break
        if not ok:raise InputError(f'finite difference failed for {names[k]}: {err}')
        J[:,k]=(-fp2+8*fp1-8*fm1+fm2)/(12*h);steps.append(h);attempts.append(a+1)
    return y,J,steps,attempts


def _validate_covariance(C):
    C=np.asarray(C,dtype=float)
    if C.ndim!=2 or C.shape[0]!=C.shape[1] or not np.isfinite(C).all(): raise InputError('covariance must be finite square')
    d=np.diag(C)
    if np.any(d<0): raise InputError('covariance has negative variance')
    pos=d>0
    if np.any(~pos) and (np.any(C[~pos,:]!=0) or np.any(C[:,~pos]!=0)): raise InputError('zero-variance row has nonzero covariance')
    if np.any(pos):
        sd=np.sqrt(d[pos]);R=C[np.ix_(pos,pos)]/np.outer(sd,sd);scale=max(1.0,float(np.max(np.abs(R))))
        if np.max(np.abs(R-R.T))>1e-12+1e-10*scale: raise InputError('covariance not symmetric on variance-normalized basis')
        R=(R+R.T)/2;ev=np.linalg.eigvalsh(R);tol=1e-12+1e-10*max(1.0,float(np.max(np.abs(ev))))
        if ev[0]<-tol: raise InputError('covariance not PSD on variance-normalized basis')
    return (C+C.T)/2

def _jac_metric(A,B,mu,cov):
    sigma=np.sqrt(np.clip(np.diag(cov),0,None));sx=np.maximum(np.abs(mu),sigma);sx=np.where(sx>0,sx,np.finfo(float).tiny)
    WA=A*sx[None,:];WB=B*sx[None,:];scale=np.maximum(np.max(np.abs(WA),axis=1),np.max(np.abs(WB),axis=1));scale=np.where(scale>0,scale,np.finfo(float).tiny)
    D=np.abs(WA-WB)/scale[:,None];return float(np.max(D)),float(np.sqrt(np.mean(D*D)))
def _hess_metric(A,B,mu,cov):
    sigma=np.sqrt(np.clip(np.diag(cov),0,None));sx=np.maximum(np.abs(mu),sigma);sx=np.where(sx>0,sx,np.finfo(float).tiny);W=np.outer(sx,sx)
    vals=[]
    for i in range(A.shape[0]):
        WA=A[i]*W;WB=B[i]*W;scale=max(float(np.max(np.abs(WA))),float(np.max(np.abs(WB))),np.finfo(float).tiny);vals.append(np.max(np.abs(WA-WB))/scale)
    return float(max(vals) if vals else 0.0)
def _cov_metric(A,B):
    da=np.clip(np.diag(A),0,None);db=np.clip(np.diag(B),0,None);s=np.sqrt(np.maximum(da,db));s=np.where(s>0,s,np.finfo(float).tiny);D=np.abs(A-B)/np.outer(s,s);return float(np.max(D)) if D.size else 0.0

def triple_verify(payload:Mapping[str,Any],*,fd_rtol:float=2e-5,ad_rtol:float=2e-10)->dict[str,Any]:
    names=list(payload.get('variables') or []);mu=np.asarray(payload.get('values'),dtype=float);C=np.asarray(payload.get('covariance'),dtype=float)
    expressions=list(payload.get('expressions') or ([payload['expression']] if 'expression' in payload else []));n=len(names)
    if not names or len(set(names))!=n or mu.shape!=(n,) or C.shape!=(n,n) or not np.isfinite(mu).all() or not expressions:raise InputError('variables/values/covariance/expressions misaligned')
    C=_validate_covariance(C)
    sv,sj,sh=_symbolic(expressions,names,mu);ar=ad.differentiate(expressions,names,mu);av=np.array(ar['values']);aj=np.array(ar['jacobian']);ah=np.array(ar['hessians']);fv,fj,steps,attempts=_finite_difference(expressions,names,mu,C)
    sa_j,_=_jac_metric(sj,aj,mu,C);af_j,_=_jac_metric(aj,fj,mu,C);sf_j,_=_jac_metric(sj,fj,mu,C);sa_h=_hess_metric(sh,ah,mu,C)
    Cs=sj@C@sj.T;Ca=aj@C@aj.T;Cf=fj@C@fj.T;covmax=max(_cov_metric(Cs,Ca),_cov_metric(Ca,Cf),_cov_metric(Cs,Cf))
    nom_scale=np.maximum.reduce([np.abs(sv),np.abs(av),np.abs(fv),np.full(len(sv),np.finfo(float).tiny)]);nom=float(np.max(np.maximum(np.abs(sv-av),np.abs(av-fv))/nom_scale))
    passed=sa_j<=ad_rtol and sa_h<=ad_rtol*10 and af_j<=fd_rtol and sf_j<=fd_rtol and covmax<=4*fd_rtol and nom<=fd_rtol
    return {'engine_version':ENGINE_VERSION,'status':'PASS' if passed else 'FAIL','method':'symbolic_vs_forward_AD_vs_5point_FD',
            'engines':['sympy_symbolic','independent_second_order_forward_AD','independent_AST_5point_finite_difference'],
            'diagnostics':{'symbolic_vs_ad_jacobian':sa_j,'symbolic_vs_ad_hessian':sa_h,'ad_vs_fd_jacobian':af_j,'symbolic_vs_fd_jacobian':sf_j,'max_pairwise_covariance_normalized_difference':covmax,'nominal_normalized_difference':nom,
                           'ad_tolerance':ad_rtol,'fd_tolerance':fd_rtol},
            'finite_difference_steps':steps,'finite_difference_attempts':attempts,'symbolic_jacobian':sj.tolist(),'ad_jacobian':aj.tolist(),'fd_jacobian':fj.tolist(),
            'symbolic_hessians':sh.tolist(),'ad_hessians':ah.tolist(),'symbolic_output_covariance':Cs.tolist(),'ad_output_covariance':Ca.tolist(),'fd_output_covariance':Cf.tolist()}
