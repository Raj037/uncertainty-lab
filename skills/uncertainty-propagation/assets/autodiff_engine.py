#!/usr/bin/env python3
"""Independent second-order forward automatic differentiation for Uncertainty Lab v0.5.

This module is deliberately independent of SymPy and finite differences.  It parses a
small closed mathematical expression language and propagates value/gradient/Hessian
jets through it.  It is the third derivative implementation used by the v0.5
three-engine acceptance gate.
"""
from __future__ import annotations
import ast, math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np

ENGINE_VERSION = "0.5.0"
MAX_EXPRESSION_CHARS = 5000

class InputError(ValueError):
    pass

@dataclass
class Jet2:
    value: float
    grad: np.ndarray
    hess: np.ndarray

    @classmethod
    def const(cls, value: float, n: int) -> "Jet2":
        return cls(float(value), np.zeros(n, dtype=float), np.zeros((n,n), dtype=float))

    def _coerce(self, other: Any) -> "Jet2":
        return other if isinstance(other, Jet2) else Jet2.const(float(other), self.grad.size)

    def __add__(self, other: Any):
        o=self._coerce(other); return Jet2(self.value+o.value, self.grad+o.grad, self.hess+o.hess)
    __radd__=__add__
    def __neg__(self): return Jet2(-self.value,-self.grad,-self.hess)
    def __sub__(self,other:Any): return self+(-self._coerce(other))
    def __rsub__(self,other:Any): return self._coerce(other)+(-self)
    def __mul__(self, other: Any):
        o=self._coerce(other)
        return Jet2(
            self.value*o.value,
            o.value*self.grad+self.value*o.grad,
            o.value*self.hess+self.value*o.hess+np.outer(self.grad,o.grad)+np.outer(o.grad,self.grad),
        )
    __rmul__=__mul__
    def reciprocal(self):
        if self.value == 0.0: raise InputError("division by zero")
        return unary(self, 1.0/self.value, -1.0/self.value**2, 2.0/self.value**3)
    def __truediv__(self,other:Any): return self*self._coerce(other).reciprocal()
    def __rtruediv__(self,other:Any): return self._coerce(other)*self.reciprocal()
    def __pow__(self, other: Any):
        if isinstance(other, Jet2):
            # General real-valued a**b implemented as exp(b*log(a)); requires a>0.
            if self.value <= 0.0: raise InputError("variable exponent requires positive base")
            return jexp(other*jlog(self))
        p=float(other)
        if self.value < 0.0 and not p.is_integer(): raise InputError("fractional power of negative base")
        if self.value == 0.0 and p < 1.0: raise InputError("power derivative undefined at zero")
        v=self.value**p
        if p == 0.0: return Jet2.const(1.0,self.grad.size)
        fp=p*(self.value**(p-1.0))
        if p == 1.0: fpp=0.0
        else:
            if self.value == 0.0 and p < 2.0: raise InputError("second derivative of power undefined at zero")
            fpp=p*(p-1.0)*(self.value**(p-2.0))
        return unary(self,v,fp,fpp)
    def __rpow__(self,other:Any):
        base=float(other)
        if base <= 0.0: raise InputError("real exponential base must be positive")
        return jexp(self*math.log(base))


def unary(x: Jet2, value: float, fp: float, fpp: float) -> Jet2:
    if not all(math.isfinite(float(v)) for v in (value,fp,fpp)):
        raise InputError("non-finite function/derivative")
    return Jet2(float(value), fp*x.grad, fp*x.hess+fpp*np.outer(x.grad,x.grad))

def jexp(x):
    v=math.exp(x.value); return unary(x,v,v,v)
def jlog(x):
    if x.value <= 0: raise InputError("log domain")
    return unary(x,math.log(x.value),1/x.value,-1/x.value**2)
def jsqrt(x):
    if x.value <= 0: raise InputError("sqrt requires positive nominal point for second derivative")
    v=math.sqrt(x.value); return unary(x,v,1/(2*v),-1/(4*v**3))
def jsin(x): return unary(x,math.sin(x.value),math.cos(x.value),-math.sin(x.value))
def jcos(x): return unary(x,math.cos(x.value),-math.sin(x.value),-math.cos(x.value))
def jtan(x):
    c=math.cos(x.value)
    if abs(c) < 1e-15: raise InputError("tan singularity")
    t=math.tan(x.value); sec2=1/(c*c); return unary(x,t,sec2,2*t*sec2)
def jasin(x):
    if abs(x.value)>=1: raise InputError("asin derivative domain")
    d=1-x.value*x.value
    return unary(x,math.asin(x.value),1/math.sqrt(d),x.value/(d**1.5))
def jacos(x):
    if abs(x.value)>=1: raise InputError("acos derivative domain")
    d=1-x.value*x.value
    return unary(x,math.acos(x.value),-1/math.sqrt(d),-x.value/(d**1.5))
def jatan(x):
    d=1+x.value*x.value
    return unary(x,math.atan(x.value),1/d,-2*x.value/(d*d))
def jsinh(x): return unary(x,math.sinh(x.value),math.cosh(x.value),math.sinh(x.value))
def jcosh(x): return unary(x,math.cosh(x.value),math.sinh(x.value),math.cosh(x.value))
def jtanh(x):
    t=math.tanh(x.value); fp=1-t*t; return unary(x,t,fp,-2*t*fp)
def jabs(x):
    if x.value == 0: raise InputError("abs is non-differentiable at zero")
    return x if x.value>0 else -x

_FUNCS={"exp":jexp,"log":jlog,"sqrt":jsqrt,"sin":jsin,"cos":jcos,"tan":jtan,
        "asin":jasin,"acos":jacos,"atan":jatan,"sinh":jsinh,"cosh":jcosh,"tanh":jtanh,"abs":jabs}

_FLOAT_FUNCS={"exp":math.exp,"log":math.log,"sqrt":math.sqrt,"sin":math.sin,"cos":math.cos,"tan":math.tan,
              "asin":math.asin,"acos":math.acos,"atan":math.atan,"sinh":math.sinh,"cosh":math.cosh,"tanh":math.tanh,"abs":abs}

def _eval(node: ast.AST, env: Mapping[str, Any], funcs: Mapping[str,Any]):
    if isinstance(node, ast.Constant) and isinstance(node.value,(int,float)): return node.value
    if isinstance(node, ast.Name):
        if node.id in env: return env[node.id]
        if node.id=="pi": return math.pi
        if node.id=="e": return math.e
        raise InputError(f"unknown name {node.id!r}")
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
        v=_eval(node.operand,env,funcs); return v if isinstance(node.op,ast.UAdd) else -v
    if isinstance(node,ast.BinOp):
        a=_eval(node.left,env,funcs); b=_eval(node.right,env,funcs)
        if isinstance(node.op,ast.Add): return a+b
        if isinstance(node.op,ast.Sub): return a-b
        if isinstance(node.op,ast.Mult): return a*b
        if isinstance(node.op,ast.Div): return a/b
        if isinstance(node.op,ast.Pow): return a**b
        raise InputError("operator not allowed")
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords and len(node.args)==1:
        fn=funcs.get(node.func.id)
        if fn is None: raise InputError(f"function not allowed: {node.func.id}")
        arg=_eval(node.args[0],env,funcs)
        if funcs is _FUNCS and not isinstance(arg,Jet2):
            n=next((v.grad.size for v in env.values() if isinstance(v,Jet2)),0)
            arg=Jet2.const(float(arg),n)
        return fn(arg)
    raise InputError(f"syntax not allowed: {type(node).__name__}")

def compile_expression(expression:str)->ast.AST:
    if not isinstance(expression,str) or not expression.strip(): raise InputError("expression must be non-empty")
    if len(expression)>MAX_EXPRESSION_CHARS: raise InputError("expression too long")
    try: tree=ast.parse(expression,mode="eval")
    except SyntaxError as e: raise InputError(f"invalid expression syntax: {e}") from e
    # Dry traversal with dummy floats occurs at evaluation; unsupported nodes fail closed.
    return tree.body

def evaluate_float(expression:str, variables:Sequence[str], values:Sequence[float])->float:
    if len(variables)!=len(values) or len(set(variables))!=len(variables): raise InputError("variables/values misaligned")
    v=float(_eval(compile_expression(expression),dict(zip(variables,map(float,values))),_FLOAT_FUNCS))
    if not math.isfinite(v): raise InputError("non-finite expression value")
    return v

def differentiate(expressions:Sequence[str]|str, variables:Sequence[str], values:Sequence[float])->dict[str,Any]:
    if isinstance(expressions,str): expressions=[expressions]
    variables=list(variables); values=np.asarray(values,dtype=float)
    n=len(variables)
    if n==0 or len(set(variables))!=n or values.shape!=(n,) or not np.isfinite(values).all():
        raise InputError("variables must be unique and values finite/aligned")
    env={}
    for i,(name,val) in enumerate(zip(variables,values)):
        g=np.zeros(n); g[i]=1.0
        env[name]=Jet2(float(val),g,np.zeros((n,n)))
    outs=[]
    for ex in expressions:
        node=compile_expression(ex)
        y=_eval(node,env,_FUNCS)
        if not isinstance(y,Jet2): y=Jet2.const(float(y),n)
        if not math.isfinite(y.value) or not np.isfinite(y.grad).all() or not np.isfinite(y.hess).all():
            raise InputError("non-finite automatic derivative")
        outs.append(y)
    return {"engine_version":ENGINE_VERSION,"method":"second_order_forward_AD","values":[o.value for o in outs],
            "jacobian":[o.grad.tolist() for o in outs],"hessians":[o.hess.tolist() for o in outs]}
