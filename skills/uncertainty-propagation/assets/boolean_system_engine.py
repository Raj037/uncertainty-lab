#!/usr/bin/env python3
"""General Boolean system reliability on shared joint input samples for v0.9.

Component failure events are defined by limit-state expressions g_i(X)<=0.
The system event is a safe Boolean expression over component names using
`and`, `or`, and `not`. All component events are evaluated on the same joint
samples, preserving common-cause/input dependence. For small systems, minimal
cut sets are derived exactly from the Boolean truth table.
"""
from __future__ import annotations
import ast, math, itertools
from typing import Any, Mapping
import numpy as np
import joint_distribution_engine as joint
ENGINE_VERSION='0.9.0'
MAX_COMPONENTS=20
MAX_CUTSET_COMPONENTS=12
class InputError(ValueError): pass


def _bool_eval(node, env):
    if isinstance(node,ast.Name):
        if node.id not in env: raise InputError(f'unknown component {node.id!r}')
        return env[node.id]
    if isinstance(node,ast.BoolOp):
        vals=[_bool_eval(x,env) for x in node.values]
        out=vals[0].copy()
        for v in vals[1:]: out=np.logical_and(out,v) if isinstance(node.op,ast.And) else np.logical_or(out,v)
        return out
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not): return np.logical_not(_bool_eval(node.operand,env))
    if isinstance(node,ast.Constant) and isinstance(node.value,bool):
        n=len(next(iter(env.values()))); return np.full(n,node.value,dtype=bool)
    raise InputError(f'unsupported Boolean syntax {type(node).__name__}')


def _compile_bool(expr):
    try: node=ast.parse(str(expr),mode='eval').body
    except SyntaxError as e: raise InputError(f'invalid system expression: {e}') from e
    # dry structural validation
    def walk(n):
        if isinstance(n,ast.Name): return
        if isinstance(n,ast.Constant) and isinstance(n.value,bool): return
        if isinstance(n,ast.BoolOp) and isinstance(n.op,(ast.And,ast.Or)):
            for x in n.values: walk(x)
            return
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.Not): walk(n.operand); return
        raise InputError(f'unsupported Boolean syntax {type(n).__name__}')
    walk(node); return node


def _truth(expr_node,names,state):
    env={n:np.array([bool(state[i])]) for i,n in enumerate(names)}
    return bool(_bool_eval(expr_node,env)[0])


def minimal_cut_sets(names, expr_node):
    d=len(names)
    if d>MAX_CUTSET_COMPONENTS: return None
    cuts=[]
    for k in range(1,d+1):
        for comb in itertools.combinations(range(d),k):
            s=[False]*d
            for i in comb:s[i]=True
            if not _truth(expr_node,names,s): continue
            minimal=True
            for j in comb:
                t=s.copy(); t[j]=False
                if _truth(expr_node,names,t): minimal=False; break
            if minimal: cuts.append([names[i] for i in comb])
    return cuts


def evaluate(payload:Mapping[str,Any])->dict[str,Any]:
    comps=dict(payload.get('components') or {}); names=list(comps)
    if not names or len(names)>MAX_COMPONENTS or len(set(names))!=len(names): raise InputError(f'components must contain 1..{MAX_COMPONENTS} unique names')
    expr=str(payload.get('system_event','')).strip()
    if not expr: raise InputError('system_event Boolean expression required')
    bnode=_compile_bool(expr)
    # Generate one shared input matrix using the established explicit joint-law engine.
    var_names,X,jmeta=joint.generate_inputs(payload)
    envX={n:X[:,i] for i,n in enumerate(var_names)}; failures={}
    for cname,gexpr in comps.items():
        node=joint._compile(str(gexpr))
        with np.errstate(all='ignore'): g=np.asarray(joint._eval(node,envX),dtype=float)
        if g.ndim==0:g=np.full(X.shape[0],float(g))
        if g.shape!=(X.shape[0],) or not np.isfinite(g).all(): raise InputError(f'component {cname} limit state produced invalid values')
        failures[cname]=g<=0
    system=_bool_eval(bnode,failures); n=len(system); pf=float(np.mean(system)); se=math.sqrt(max(pf*(1-pf),0)/n)
    comp_pf={k:float(np.mean(v)) for k,v in failures.items()}
    F=np.column_stack([failures[n].astype(float) for n in names])
    if len(names)>1:
        corr=np.corrcoef(F,rowvar=False)
        corr=np.asarray(corr,dtype=float)
        corr_json=[[None if not np.isfinite(x) else float(x) for x in row] for row in corr]
    else:corr_json=[[1.0]]
    cuts=minimal_cut_sets(names,bnode) if bool(payload.get('derive_minimal_cut_sets',True)) else None
    return {'engine_version':ENGINE_VERSION,'kind':'boolean_system_reliability','status':'PASS','system_event':expr,'component_order':names,'component_failure_probability':comp_pf,
            'system_failure_probability':pf,'standard_error':se,'n':n,'joint_model':jmeta,'component_failure_correlation':corr_json,'minimal_cut_sets':cuts,
            'claim_boundary':'System probability is the direct joint-sample event probability. Minimal cut sets describe Boolean logic, not statistical independence or causal structure.'}
