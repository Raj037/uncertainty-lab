#!/usr/bin/env python3
"""Nonlinear complex-valued uncertainty propagation for Uncertainty Lab v1.0.

Uses an explicitly ordered real/imag multivariate-normal input model and safe
vectorized complex expression evaluation. Returns Monte Carlo real/imag covariance,
Hermitian covariance, pseudo-covariance, magnitude and circular-phase summaries,
plus a local real-coordinate finite-difference linearization for comparison.
"""
from __future__ import annotations
import ast, math
from typing import Any, Mapping
import numpy as np
import joint_distribution_engine as joint
ENGINE_VERSION='1.0.0'
MAX_COMPLEX_VARIABLES=32
MAX_SAMPLES=1_000_000
class InputError(ValueError):pass
_CF={'sqrt':np.sqrt,'exp':np.exp,'log':np.log,'sin':np.sin,'cos':np.cos,'tan':np.tan,'sinh':np.sinh,'cosh':np.cosh,'tanh':np.tanh,'abs':np.abs,'conj':np.conj,'real':np.real,'imag':np.imag}

def _eval(node,env):
    if isinstance(node,ast.Constant) and isinstance(node.value,(int,float,complex)):return node.value
    if isinstance(node,ast.Name):
        if node.id in env:return env[node.id]
        if node.id=='pi':return math.pi
        if node.id=='e':return math.e
        if node.id=='j':return 1j
        raise InputError(f'unknown name {node.id!r}')
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
        v=_eval(node.operand,env);return v if isinstance(node.op,ast.UAdd) else -v
    if isinstance(node,ast.BinOp):
        a=_eval(node.left,env);b=_eval(node.right,env)
        if isinstance(node.op,ast.Add):return a+b
        if isinstance(node.op,ast.Sub):return a-b
        if isinstance(node.op,ast.Mult):return a*b
        if isinstance(node.op,ast.Div):return a/b
        if isinstance(node.op,ast.Pow):return a**b
        raise InputError('operator not allowed')
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords and len(node.args)==1:
        fn=_CF.get(node.func.id)
        if fn is None:raise InputError(f'function not allowed {node.func.id!r}')
        return fn(_eval(node.args[0],env))
    raise InputError(f'complex expression syntax not allowed: {type(node).__name__}')
def _compile(s):
    if not isinstance(s,str) or not s.strip() or len(s)>5000:raise InputError('invalid complex expression')
    try:n=ast.parse(s,mode='eval').body
    except SyntaxError as e:raise InputError(str(e)) from e
    # dry validation by recursive syntax shape with symbolic zero env names deferred
    return n

def _c(v):
    if isinstance(v,(int,float,complex)):return complex(v)
    if isinstance(v,Mapping):return complex(float(v.get('re',0)),float(v.get('im',0)))
    raise InputError('complex value must be number or {re,im}')
def _obj(z):return {'re':float(np.real(z)),'im':float(np.imag(z))}

def _complex_covariances_from_real(C,n):
    K=np.zeros((n,n),dtype=complex);P=np.zeros((n,n),dtype=complex)
    for i in range(n):
        xi,yi=2*i,2*i+1
        for j in range(n):
            xj,yj=2*j,2*j+1
            K[i,j]=C[xi,xj]+C[yi,yj]+1j*(C[yi,xj]-C[xi,yj])
            P[i,j]=C[xi,xj]-C[yi,yj]+1j*(C[xi,yj]+C[yi,xj])
    return K,P

def propagate(payload:Mapping[str,Any])->dict[str,Any]:
    names=list(payload.get('variables') or []);vals=list(payload.get('values') or []);exprs=list(payload.get('expressions') or ([payload['expression']] if 'expression' in payload else []));n=int(payload.get('n',100000));seed=int(payload.get('seed',1729))
    if not names or len(names)!=len(vals) or len(set(names))!=len(names) or len(names)>MAX_COMPLEX_VARIABLES or not exprs or n<1000 or n>MAX_SAMPLES:raise InputError('complex propagation input mismatch')
    mu_c=np.array([_c(v) for v in vals],dtype=complex);mu=np.empty(2*len(names));mu[0::2]=mu_c.real;mu[1::2]=mu_c.imag
    C=joint.validate_covariance(payload.get('covariance'),'complex real-im covariance')
    if C.shape!=(2*len(names),2*len(names)):raise InputError('complex covariance must be 2N x 2N in [Re,Im] order')
    nodes=[_compile(e) for e in exprs]
    def evaluate(realmat):
        env={names[i]:realmat[...,2*i]+1j*realmat[...,2*i+1] for i in range(len(names))};outs=[]
        for node in nodes:
            with np.errstate(all='ignore'):z=np.asarray(_eval(node,env))
            if z.ndim==0:z=np.full(realmat.shape[0] if realmat.ndim==2 else 1,z,dtype=complex)
            outs.append(z)
        return outs
    nominal=evaluate(mu[None,:]);nom=np.array([z[0] for z in nominal],dtype=complex)
    rng=np.random.Generator(np.random.PCG64(seed));X=rng.multivariate_normal(mu,C,size=n,check_valid='raise');outs=evaluate(X);valid=np.ones(n,dtype=bool)
    for z in outs:valid &= np.isfinite(z.real)&np.isfinite(z.imag)
    invalid=int(np.sum(~valid))
    if invalid==n:return {'engine_version':ENGINE_VERSION,'kind':'nonlinear_complex_monte_carlo','status':'BLOCKED_ALL_SAMPLES_INVALID','invalid_fraction':1.0}
    Z=np.column_stack([z[valid] for z in outs]);R=np.empty((np.sum(valid),2*len(exprs)));R[:,0::2]=Z.real;R[:,1::2]=Z.imag;Cr=np.atleast_2d(np.cov(R,rowvar=False,ddof=1));mean=np.mean(Z,axis=0)
    K,P=_complex_covariances_from_real(Cr,len(exprs))
    mag=[];phase=[]
    for j in range(len(exprs)):
        a=np.abs(Z[:,j]);ph=np.angle(Z[:,j]);cmean=np.mean(np.exp(1j*ph));mag.append({'mean':float(a.mean()),'std':float(a.std(ddof=1)),'quantiles':[float(x) for x in np.quantile(a,[.05,.5,.95])]});phase.append({'circular_mean_rad':float(np.angle(cmean)),'resultant_length':float(abs(cmean))})
    # Local real-coordinate 5-point Jacobian around nominal input.
    m=2*len(exprs);d=2*len(names);J=np.zeros((m,d));eps=np.finfo(float).eps**.2
    for k in range(d):
        char=max(abs(float(mu[k])),math.sqrt(max(float(C[k,k]),0.0)),1.0);h=eps*char;e=np.zeros(d);e[k]=h
        pts=[]
        for fac in [2,1,-1,-2]:
            zz=evaluate((mu+fac*e)[None,:]);rv=np.empty(m);q=np.array([z[0] for z in zz]);rv[0::2]=q.real;rv[1::2]=q.imag;pts.append(rv)
        J[:,k]=(-pts[0]+8*pts[1]-8*pts[2]+pts[3])/(12*h)
    Clin=J@C@J.T;Clin=(Clin+Clin.T)/2
    return {'engine_version':ENGINE_VERSION,'kind':'nonlinear_complex_monte_carlo','status':'PASS' if invalid==0 else 'DOMAIN_VIOLATIONS','variable_order':[x for nm in names for x in (f'Re({nm})',f'Im({nm})')],'expressions':exprs,'nominal':[_obj(z) for z in nom],'mean':[_obj(z) for z in mean],
            'output_real_im_covariance':Cr.tolist(),'complex_covariance_E_dz_dzH':[[ _obj(z) for z in row] for row in K],'pseudo_covariance_E_dz_dzT':[[ _obj(z) for z in row] for row in P],
            'magnitude':mag,'phase':phase,'invalid_count':invalid,'invalid_fraction':invalid/n,'n':n,'seed':seed,'local_linearized_real_im_covariance':Clin.tolist(),
            'claim_boundary':'Monte Carlo assumes the supplied real/imag input vector is multivariate normal. Phase summaries are circular and may be uninformative when resultant length is small. Local linearization is diagnostic, not a replacement for nonlinear propagation.'}
