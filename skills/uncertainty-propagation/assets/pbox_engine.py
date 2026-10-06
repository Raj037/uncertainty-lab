#!/usr/bin/env python3
"""Parametric probability-box / epistemic distribution envelopes for v0.9.

The engine distinguishes aleatory variation inside each candidate distribution
from epistemic uncertainty about distribution parameters. It supports analytic/
numerically optimized CDF envelopes for selected scalar parametric families and
sampled propagation over an explicit parameter-family grid.

The propagated family envelope is an envelope over the explicitly evaluated
parameter scenarios; it is not advertised as a proof of the global extremum for
an arbitrary nonlinear multivariate model.
"""
from __future__ import annotations
import itertools, math
from typing import Any, Mapping
import numpy as np
from scipy import optimize, stats
from scipy.stats import qmc
import joint_distribution_engine as joint

ENGINE_VERSION='0.9.0'
MAX_SCENARIOS=4096
MAX_SAMPLES=524288
class InputError(ValueError): pass


def _interval(v,name,positive=False):
    a=np.asarray(v,dtype=float)
    if a.shape!=(2,) or not np.isfinite(a).all() or a[0]>a[1]: raise InputError(f'{name} must be [lower,upper]')
    if positive and a[0]<=0: raise InputError(f'{name} must be strictly positive')
    return float(a[0]),float(a[1])


def _family_cdf(kind,x,params):
    if kind=='normal': return float(stats.norm.cdf(x,loc=params[0],scale=params[1]))
    if kind=='lognormal':
        if x<=0:return 0.0
        return float(stats.lognorm.cdf(x,s=params[1],scale=math.exp(params[0])))
    if kind=='uniform':
        lo,hi=params
        if x<=lo:return 0.0
        if x>=hi:return 1.0
        return float((x-lo)/(hi-lo))
    raise InputError(f'unsupported p-box family {kind!r}')


def cdf_bounds(spec:Mapping[str,Any], x:float)->dict[str,Any]:
    kind=str(spec.get('kind','')).lower(); x=float(x)
    if kind=='normal':
        m=_interval(spec.get('mean'), 'mean'); s=_interval(spec.get('std'),'std',True); bounds=[m,s]
    elif kind=='lognormal':
        m=_interval(spec.get('log_mean'),'log_mean'); s=_interval(spec.get('log_std'),'log_std',True); bounds=[m,s]
    elif kind=='uniform':
        lo=_interval(spec.get('lower'),'lower'); hi=_interval(spec.get('upper'),'upper')
        # all admissible pairs must satisfy lower < upper
        if lo[1]>=hi[0]: raise InputError('uniform p-box requires max(lower) < min(upper)')
        bounds=[lo,hi]
    else: raise InputError('kind must be normal, lognormal, or uniform')
    def f(v): return _family_cdf(kind,x,v)
    # Small deterministic global optimization on a bounded 2-D rectangle.
    de_lo=optimize.differential_evolution(f,bounds,seed=1729,polish=True,tol=1e-10,workers=1)
    de_hi=optimize.differential_evolution(lambda z:-f(z),bounds,seed=1729,polish=True,tol=1e-10,workers=1)
    lower=max(0.0,min(1.0,float(de_lo.fun))); upper=max(0.0,min(1.0,float(-de_hi.fun)))
    return {'engine_version':ENGINE_VERSION,'kind':'parametric_pbox_cdf','family':kind,'x':x,'cdf_lower':lower,'cdf_upper':upper,
            'lower_extremizer':[float(v) for v in de_lo.x],'upper_extremizer':[float(v) for v in de_hi.x],
            'claim_boundary':'Bounds are optimized over the declared parametric family; they do not include distributions outside that family.'}


def probability_interval(spec:Mapping[str,Any], lower:float|None=None, upper:float|None=None)->dict[str,Any]:
    if lower is None and upper is None: raise InputError('at least one interval bound required')
    if lower is not None and upper is not None and not float(lower)<float(upper): raise InputError('lower must be < upper')
    if lower is None:
        b=cdf_bounds(spec,float(upper)); pl,pu=b['cdf_lower'],b['cdf_upper']
    elif upper is None:
        a=cdf_bounds(spec,float(lower)); pl,pu=1-a['cdf_upper'],1-a['cdf_lower']
    else:
        a=cdf_bounds(spec,float(lower)); b=cdf_bounds(spec,float(upper))
        # Conservative p-box interval probability envelope. This respects the CDF
        # bounds without claiming the same parameter point realizes both extrema.
        pl=max(0.0,b['cdf_lower']-a['cdf_upper']); pu=min(1.0,b['cdf_upper']-a['cdf_lower'])
    return {'engine_version':ENGINE_VERSION,'kind':'pbox_probability_interval','probability_lower':float(pl),'probability_upper':float(pu),
            'claim_boundary':'Two-sided interval bounds are conservative consequences of the marginal CDF envelope and may not be tight for a shared parametric family.'}


def _grid(lo,hi,n):
    if n<2:return np.array([(lo+hi)/2])
    return np.linspace(lo,hi,n)


def _marginal_scenarios(spec,grid_points):
    kind=str(spec.get('kind','')).lower()
    if kind=='normal':
        m=_interval(spec['mean'],'mean'); s=_interval(spec['std'],'std',True)
        return [{'kind':'normal','mean':float(mu),'std':float(sd)} for mu in _grid(*m,grid_points) for sd in _grid(*s,grid_points)]
    if kind=='lognormal':
        m=_interval(spec['log_mean'],'log_mean'); s=_interval(spec['log_std'],'log_std',True)
        return [{'kind':'lognormal','log_mean':float(mu),'log_std':float(sd)} for mu in _grid(*m,grid_points) for sd in _grid(*s,grid_points)]
    if kind=='uniform':
        lo=_interval(spec['lower'],'lower'); hi=_interval(spec['upper'],'upper')
        out=[]
        for a in _grid(*lo,grid_points):
            for b in _grid(*hi,grid_points):
                if a<b: out.append({'kind':'uniform','lower':float(a),'upper':float(b)})
        if not out: raise InputError('uniform family has no valid parameter scenarios')
        return out
    if kind=='fixed': return [dict(spec['marginal'])]
    raise InputError(f'unsupported propagated p-box family {kind!r}')


def propagate(payload:Mapping[str,Any])->dict[str,Any]:
    names=list(payload.get('variables') or []); families=list(payload.get('families') or []); expr=str(payload.get('expression','')).strip()
    if not names or len(names)!=len(families) or len(set(names))!=len(names) or not expr: raise InputError('variables/families/expression mismatch')
    gp=int(payload.get('grid_points',2)); n=int(payload.get('n',16384)); seed=int(payload.get('seed',1729)); thresholds=[float(x) for x in payload.get('thresholds',[])]
    if gp<2 or gp>7 or n<1024 or n>MAX_SAMPLES: raise InputError('invalid grid_points or sample count')
    grids=[_marginal_scenarios(f,gp) for f in families]
    total=math.prod(len(g) for g in grids)
    if total>MAX_SCENARIOS: raise InputError(f'epistemic scenario count {total} exceeds {MAX_SCENARIOS}')
    # Common randomized Sobol uniforms reduce scenario-comparison noise.
    m=int(math.ceil(math.log2(n))); U=qmc.Sobol(len(names),scramble=True,seed=seed).random_base2(m)[:n]
    node=joint._compile(expr); scenario=[]
    qs=[float(q) for q in payload.get('quantiles',[.05,.5,.95])]
    if any(not 0<q<1 for q in qs): raise InputError('quantiles must lie in (0,1)')
    for combo in itertools.product(*grids):
        X=np.column_stack([joint._marginal_ppf(U[:,i],combo[i]) for i in range(len(names))]); env={nm:X[:,i] for i,nm in enumerate(names)}
        with np.errstate(all='ignore'):Y=np.asarray(joint._eval(node,env),dtype=float)
        if Y.ndim==0:Y=np.full(n,float(Y))
        if Y.shape!=(n,) or not np.isfinite(Y).all(): raise InputError('expression produced invalid values in an epistemic scenario')
        scenario.append({'marginals':list(combo),'mean':float(Y.mean()),'std':float(Y.std(ddof=1)),'quantiles':[float(x) for x in np.quantile(Y,qs)],
                         'threshold_cdf':[float(np.mean(Y<=t)) for t in thresholds]})
    means=np.array([s['mean'] for s in scenario]); stds=np.array([s['std'] for s in scenario]); qmat=np.array([s['quantiles'] for s in scenario])
    out={'engine_version':ENGINE_VERSION,'kind':'sampled_parametric_pbox_propagation','status':'PASS','scenario_count':len(scenario),'sample_count_per_scenario':n,
         'mean_lower':float(means.min()),'mean_upper':float(means.max()),'std_lower':float(stds.min()),'std_upper':float(stds.max()),
         'quantile_probabilities':qs,'quantile_lower':qmat.min(0).tolist(),'quantile_upper':qmat.max(0).tolist(),
         'claim_boundary':'Envelope is complete for the explicitly evaluated parameter grid. For nonlinear models, unexamined interior parameter values may be more extreme unless separately proven otherwise.'}
    if thresholds:
        C=np.array([s['threshold_cdf'] for s in scenario]); out['thresholds']=thresholds; out['cdf_lower']=C.min(0).tolist(); out['cdf_upper']=C.max(0).tolist()
    if bool(payload.get('return_scenarios',False)): out['scenarios']=scenario
    return out
