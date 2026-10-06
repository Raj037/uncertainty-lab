#!/usr/bin/env python3
"""Conformity assessment and measurement-decision risk for Uncertainty Lab v0.7.

Implements scalar tolerance/acceptance intervals, item-specific conformance probability,
guarded acceptance design for a normal measurement-result distribution, and global
consumer/producer risk simulation. The engine reports risk; it does not choose a legal,
regulatory, or contractual decision rule on the user's behalf.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from scipy import optimize, stats

import joint_distribution_engine as joint

ENGINE_VERSION = "0.7.0"
MAX_SAMPLES = 2_000_000


class InputError(ValueError):
    pass


def _limits(payload: Mapping[str, Any], prefix: str = "tolerance"):
    lo = payload.get(f"{prefix}_lower")
    hi = payload.get(f"{prefix}_upper")
    lo = -math.inf if lo is None else float(lo)
    hi = math.inf if hi is None else float(hi)
    if not math.isfinite(lo) and not math.isfinite(hi):
        raise InputError(f"at least one {prefix} limit is required")
    if not lo < hi:
        raise InputError(f"{prefix}_lower must be < {prefix}_upper")
    return lo, hi


def _normal_conformance(mean: float, sd: float, lo: float, hi: float) -> float:
    if not math.isfinite(mean) or not math.isfinite(sd) or sd <= 0:
        raise InputError("normal measurement result requires finite mean and positive standard uncertainty")
    plo = 0.0 if lo == -math.inf else float(stats.norm.cdf((lo-mean)/sd))
    phi = 1.0 if hi == math.inf else float(stats.norm.cdf((hi-mean)/sd))
    return max(0.0, min(1.0, phi-plo))


def conformance_probability(payload: Mapping[str, Any]) -> dict[str, Any]:
    lo, hi = _limits(payload, "tolerance")
    dist = dict(payload.get("measurement_distribution") or {})
    kind = str(dist.get("kind", "normal")).lower()
    if kind in {"normal", "gaussian"}:
        mean=float(dist.get("mean"));sd=float(dist.get("std",dist.get("standard_uncertainty")))
        pc=_normal_conformance(mean,sd,lo,hi);summary={"kind":"normal","mean":mean,"std":sd}
    elif kind in {"samples","empirical"}:
        x=np.asarray(dist.get("values"),dtype=float)
        if x.ndim!=1 or x.size<20 or x.size>MAX_SAMPLES or not np.isfinite(x).all():raise InputError('empirical measurement distribution requires 20..MAX_SAMPLES finite values')
        pc=float(np.mean((x>=lo)&(x<=hi)));summary={"kind":"empirical","n":int(x.size)}
    else:raise InputError("measurement_distribution.kind must be normal or empirical samples")
    return {"engine_version":ENGINE_VERSION,"conformance_probability":pc,"nonconformance_probability":1-pc,
            "tolerance_interval":[None if lo==-math.inf else lo,None if hi==math.inf else hi],"measurement_distribution":summary,
            "claim_boundary":"This is probability under the supplied distribution for the measurand. It is not a frequentist coverage probability unless the supplied distribution has that interpretation."}


def decide(payload: Mapping[str, Any]) -> dict[str, Any]:
    tl,tu=_limits(payload,'tolerance');al,au=_limits(payload,'acceptance');y=float(payload.get('measured_value'))
    if not math.isfinite(y):raise InputError('measured_value must be finite')
    if al<tl or au>tu:
        # Expanded acceptance may be intentional but is not guarded acceptance; require explicit opt-in.
        if not bool(payload.get('allow_expanded_acceptance',False)):raise InputError('acceptance interval extends outside tolerance interval; explicit allow_expanded_acceptance=true required')
    accept=bool(al<=y<=au);prob=conformance_probability(payload);pc=float(prob['conformance_probability'])
    return {"engine_version":ENGINE_VERSION,"decision":"ACCEPT" if accept else "REJECT","measured_value":y,
            "acceptance_interval":[None if al==-math.inf else al,None if au==math.inf else au],"tolerance_interval":[None if tl==-math.inf else tl,None if tu==math.inf else tu],
            "conformance_probability":pc,"specific_consumer_risk_if_accepted":1-pc if accept else None,
            "specific_producer_risk_if_rejected":pc if not accept else None,
            "claim_boundary":"Risk is conditional on the supplied measurement-result distribution and decision rule; the engine does not determine whether the rule is legally or contractually acceptable."}


def design_guard_band(payload: Mapping[str, Any]) -> dict[str, Any]:
    tl,tu=_limits(payload,'tolerance');sd=float(payload.get('standard_uncertainty'));alpha=float(payload.get('max_specific_consumer_risk',0.025))
    if not math.isfinite(sd) or sd<=0 or not 0<alpha<0.5:raise InputError('standard_uncertainty must be >0 and target risk in (0,0.5)')
    finite_lo=math.isfinite(tl);finite_hi=math.isfinite(tu)
    def risk_at(mean):return 1-_normal_conformance(mean,sd,tl,tu)
    if finite_lo and finite_hi:
        half=(tu-tl)/2
        def worst(w):return max(risk_at(tl+w),risk_at(tu-w))
        if worst(half)>alpha:
            return {"engine_version":ENGINE_VERSION,"status":"NO_NONEMPTY_ACCEPTANCE_INTERVAL_MEETS_TARGET","target_consumer_risk":alpha}
        if worst(0)<=alpha:w=0.0
        else:w=float(optimize.brentq(lambda z:worst(z)-alpha,0,half,xtol=1e-14))
        al,au=tl+w,tu-w;wr=worst(w)
    elif finite_hi:
        # risk at upper acceptance boundary is P(X>TU)
        w=max(0.0,float(stats.norm.ppf(1-alpha))*sd);al=-math.inf;au=tu-w;wr=risk_at(au)
    else:
        w=max(0.0,float(stats.norm.ppf(1-alpha))*sd);al=tl+w;au=math.inf;wr=risk_at(al)
    return {"engine_version":ENGINE_VERSION,"status":"PASS","guard_band_width":w,
            "acceptance_lower":None if al==-math.inf else al,"acceptance_upper":None if au==math.inf else au,
            "tolerance_lower":None if tl==-math.inf else tl,"tolerance_upper":None if tu==math.inf else tu,
            "target_max_specific_consumer_risk":alpha,"consumer_risk_at_active_acceptance_boundary":wr,
            "claim_boundary":"Guard-band design assumes a normal measurand distribution centered at a candidate measured value with known standard uncertainty. Different priors/asymmetric loss functions require a different decision model."}


def global_risk(payload: Mapping[str, Any]) -> dict[str, Any]:
    tl,tu=_limits(payload,'tolerance');al,au=_limits(payload,'acceptance');n=int(payload.get('n',200000));seed=int(payload.get('seed',1729))
    if n<1000 or n>MAX_SAMPLES:raise InputError(f'n must lie in [1000,{MAX_SAMPLES}]')
    true_spec=dict(payload.get('true_distribution') or {});err_spec=dict(payload.get('measurement_error_distribution') or {})
    if not true_spec or not err_spec:raise InputError('explicit true_distribution and measurement_error_distribution required for global risk')
    rng=np.random.Generator(np.random.PCG64(seed));u=rng.random((n,2))
    try:x=joint._marginal_ppf(u[:,0],true_spec);e=joint._marginal_ppf(u[:,1],err_spec)
    except Exception as exc:raise InputError(str(exc)) from exc
    y=x+e;conf=(x>=tl)&(x<=tu);accept=(y>=al)&(y<=au)
    fa=accept&(~conf);fr=(~accept)&conf
    pa=float(np.mean(accept));pr=1-pa
    return {"engine_version":ENGINE_VERSION,"method":"global_risk_monte_carlo","n":n,"seed":seed,
            "global_consumer_risk_joint_probability_false_accept":float(np.mean(fa)),
            "global_producer_risk_joint_probability_false_reject":float(np.mean(fr)),
            "conditional_consumer_risk_given_accept":None if pa==0 else float(np.mean(fa)/pa),
            "conditional_producer_risk_given_reject":None if pr==0 else float(np.mean(fr)/pr),
            "acceptance_probability":pa,"true_conformance_probability":float(np.mean(conf)),
            "claim_boundary":"Global risks depend on the supplied population distribution of true item values and measurement-error model. They are not item-specific risks unless those distributions are justified for the item/population at hand."}
