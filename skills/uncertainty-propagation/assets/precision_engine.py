#!/usr/bin/env python3
"""High-precision and interval numerical assurance for Uncertainty Lab v0.6.

This is a numerical-assurance layer, not a new scientific model. It re-evaluates the same
closed expression language at arbitrary precision, computes a high-precision symbolic
Jacobian/covariance, and optionally encloses expression values over explicit input boxes
using interval arithmetic. A float/high-precision disagreement is never silently repaired.
"""
from __future__ import annotations
import ast, math
from typing import Any, Mapping, Sequence
import numpy as np
import mpmath as mp
import sympy as sp
import autodiff_engine as ad

ENGINE_VERSION='0.6.0'
MAX_DPS=300
class InputError(ValueError): pass

_SPF={'sqrt':sp.sqrt,'exp':sp.exp,'log':sp.log,'sin':sp.sin,'cos':sp.cos,'tan':sp.tan,'asin':sp.asin,'acos':sp.acos,'atan':sp.atan,'sinh':sp.sinh,'cosh':sp.cosh,'tanh':sp.tanh,'abs':sp.Abs}
_IVF={'sqrt':mp.iv.sqrt,'exp':mp.iv.exp,'log':mp.iv.log,'sin':mp.iv.sin,'cos':mp.iv.cos,'tan':mp.iv.tan,'sinh':lambda x:(mp.iv.exp(x)-mp.iv.exp(-x))/2,'cosh':lambda x:(mp.iv.exp(x)+mp.iv.exp(-x))/2,'tanh':lambda x:(mp.iv.exp(2*x)-1)/(mp.iv.exp(2*x)+1),'abs':abs}

def _sym(node, env):
    if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)): return sp.Rational(str(node.value)) if isinstance(node.value,float) else sp.Integer(node.value)
    if isinstance(node,ast.Name):
        if node.id in env:return env[node.id]
        if node.id=='pi':return sp.pi
        if node.id=='e':return sp.E
        raise InputError(f'unknown name {node.id!r}')
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
        v=_sym(node.operand,env);return v if isinstance(node.op,ast.UAdd) else -v
    if isinstance(node,ast.BinOp):
        a,b=_sym(node.left,env),_sym(node.right,env)
        if isinstance(node.op,ast.Add):return a+b
        if isinstance(node.op,ast.Sub):return a-b
        if isinstance(node.op,ast.Mult):return a*b
        if isinstance(node.op,ast.Div):return a/b
        if isinstance(node.op,ast.Pow):return a**b
        raise InputError('operator not allowed')
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords and len(node.args)==1:
        f=_SPF.get(node.func.id)
        if f is None:raise InputError(f'function not allowed: {node.func.id}')
        return f(_sym(node.args[0],env))
    raise InputError(f'syntax not allowed: {type(node).__name__}')

def _iv(node, env):
    if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)):return mp.iv.mpf(str(node.value))
    if isinstance(node,ast.Name):
        if node.id in env:return env[node.id]
        if node.id=='pi':return mp.iv.pi
        if node.id=='e':return mp.iv.e
        raise InputError(f'unknown name {node.id!r}')
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
        v=_iv(node.operand,env);return v if isinstance(node.op,ast.UAdd) else -v
    if isinstance(node,ast.BinOp):
        a,b=_iv(node.left,env),_iv(node.right,env)
        if isinstance(node.op,ast.Add):return a+b
        if isinstance(node.op,ast.Sub):return a-b
        if isinstance(node.op,ast.Mult):return a*b
        if isinstance(node.op,ast.Div):return a/b
        if isinstance(node.op,ast.Pow):
            # Interval exponent must be a numeric constant to avoid ambiguous complex branches.
            if not isinstance(node.right,ast.Constant) or not isinstance(node.right.value,(int,float)):raise InputError('interval power exponent must be numeric constant')
            return a**node.right.value
        raise InputError('operator not allowed')
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords and len(node.args)==1:
        f=_IVF.get(node.func.id)
        if f is None:raise InputError(f'interval function not supported: {node.func.id}')
        return f(_iv(node.args[0],env))
    raise InputError(f'syntax not allowed: {type(node).__name__}')

def _node(expr): return ad.compile_expression(expr)

def high_precision_propagate(payload:Mapping[str,Any], dps:int=80)->dict[str,Any]:
    dps=int(dps)
    if dps<30 or dps>MAX_DPS:raise InputError(f'dps must be in [30,{MAX_DPS}]')
    names=list(payload.get('variables') or []); vals=list(payload.get('values') or []); exprs=list(payload.get('expressions') or ([payload['expression']] if 'expression' in payload else []))
    if not names or len(names)!=len(vals) or len(set(names))!=len(names) or not exprs:raise InputError('variables/values/expressions misaligned')
    cov=np.asarray(payload.get('covariance'),dtype=float)
    if cov.shape!=(len(names),len(names)) or not np.isfinite(cov).all():raise InputError('finite covariance with matching dimension required')
    syms=[sp.Symbol(n,real=True) for n in names]; env=dict(zip(names,syms)); sexpr=[_sym(_node(e),env) for e in exprs]
    subs={s:sp.Float(repr(float(v)),dps) for s,v in zip(syms,vals)}
    with mp.workdps(dps):
        y=[sp.N(e.subs(subs),dps) for e in sexpr]
        J=sp.Matrix(sexpr).jacobian(syms)
        Jn=J.subs(subs).evalf(dps)
        # Convert through decimal strings to mpmath, avoiding an early binary-float round trip.
        Jm=mp.matrix([[mp.mpf(str(Jn[i,j])) for j in range(len(names))] for i in range(len(exprs))])
        Cm=mp.matrix([[mp.mpf(repr(float(cov[i,j]))) for j in range(len(names))] for i in range(len(names))])
        Oy=Jm*Cm*Jm.T
        hp_y=[mp.mpf(str(v)) for v in y]
        hp_cov=[[Oy[i,j] for j in range(len(exprs))] for i in range(len(exprs))]
        # Float path is deliberately separate and may lose cancellation-sensitive digits.
        f=ad.differentiate(exprs,names,vals); fy=np.asarray(f['values'],dtype=float); fJ=np.asarray(f['jacobian'],dtype=float); fc=fJ@cov@fJ.T
        val_rel=[]
        for a,b in zip(fy,hp_y):
            scale=max(abs(b),mp.mpf('1e-'+str(dps-5))); val_rel.append(float(abs(mp.mpf(repr(float(a)))-b)/scale))
        cov_rel=[]
        for i in range(len(exprs)):
            for j in range(len(exprs)):
                b=hp_cov[i][j]; scale=max(abs(b),mp.mpf('1e-'+str(dps-5))); cov_rel.append(float(abs(mp.mpf(repr(float(fc[i,j])))-b)/scale))
    tol=float(payload.get('precision_relative_tolerance',1e-10))
    if tol<0 or not math.isfinite(tol):raise InputError('precision_relative_tolerance must be finite nonnegative')
    maxv=max(val_rel,default=0.0);maxc=max(cov_rel,default=0.0);agree=max(maxv,maxc)<=tol
    return {'engine_version':ENGINE_VERSION,'method':'arbitrary_precision_symbolic_Jacobian','dps':dps,'status':'PASS' if agree else 'FLOAT_DISAGREEMENT','float_high_precision_relative_tolerance':tol,'max_nominal_relative_disagreement':maxv,'max_covariance_relative_disagreement':maxc,
            'high_precision_nominal':[mp.nstr(v,dps) for v in hp_y],'high_precision_output_covariance':[[mp.nstr(v,dps) for v in row] for row in hp_cov],
            'float_nominal':fy.tolist(),'float_output_covariance':fc.tolist(),'claim_boundary':'A high-precision PASS checks numerical agreement for this expression/covariance. It does not validate the scientific model.'}

def interval_evaluate(payload:Mapping[str,Any])->dict[str,Any]:
    names=list(payload.get('variables') or []); exprs=list(payload.get('expressions') or ([payload['expression']] if 'expression' in payload else [])); bounds=payload.get('interval_bounds')
    if not names or not exprs or not isinstance(bounds,Mapping):raise InputError('variables, expression(s), and interval_bounds mapping required')
    env={}
    for n in names:
        b=bounds.get(n)
        if not isinstance(b,(list,tuple)) or len(b)!=2:raise InputError(f'interval_bounds[{n!r}] must be [lower,upper]')
        lo,hi=map(float,b)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo>hi:raise InputError('invalid interval bound')
        env[n]=mp.iv.mpf([lo,hi])
    out=[]
    for e in exprs:
        try:v=_iv(_node(e),env)
        except Exception as exc:raise InputError(f'interval evaluation failed: {exc}') from exc
        # mpmath interval endpoints are intervals themselves; convert via strings robustly.
        lo=float(v.a);hi=float(v.b);out.append([lo,hi])
    return {'engine_version':ENGINE_VERSION,'method':'interval_box_enclosure','status':'ENCLOSED','intervals':out,'bounds':{k:list(map(float,bounds[k])) for k in names},'claim_boundary':'The interval encloses expression values over the supplied input box; it is not a probability or coverage interval.'}

def verify(payload:Mapping[str,Any])->dict[str,Any]:
    hp=high_precision_propagate(payload,int(payload.get('dps',80)))
    iv=None
    if 'interval_bounds' in payload:iv=interval_evaluate(payload)
    return {'engine_version':ENGINE_VERSION,'status':hp['status'],'high_precision':hp,'interval':iv}
