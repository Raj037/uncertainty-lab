#!/usr/bin/env python3
"""Rare-event and structural-reliability methods for Uncertainty Lab v0.7.

FORM/SORM are implemented for an explicitly supplied nonsingular multivariate-normal
input model. Importance sampling and subset simulation operate in the same standard-
normal space. No non-normal isoprobabilistic transform is inferred automatically.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from scipy import optimize, stats

import autodiff_engine as ad
import joint_distribution_engine as joint

ENGINE_VERSION = "0.7.0"
MAX_DIM = 64
MAX_SAMPLES = 2_000_000


class InputError(ValueError):
    pass


def _normal_model(payload: Mapping[str, Any]):
    names = list(payload.get("variables") or [])
    d = len(names)
    if not 1 <= d <= MAX_DIM or len(set(names)) != d:
        raise InputError(f"variables must contain 1..{MAX_DIM} unique names")
    mean = np.asarray(payload.get("mean"), dtype=float)
    cov = np.asarray(payload.get("covariance"), dtype=float)
    if mean.shape != (d,) or cov.shape != (d, d) or not np.isfinite(mean).all():
        raise InputError("mean/covariance dimensions mismatch")
    try:
        cov = joint.validate_covariance(cov, "reliability covariance")
    except Exception as exc:
        raise InputError(str(exc)) from exc
    ev, q = np.linalg.eigh(cov)
    scale = float(np.max(np.abs(ev))) if ev.size else 0.0
    tol = (1e-12 + 1e-10) * scale if scale > 0 else 0.0
    if np.any(ev <= tol):
        raise InputError("FORM/SORM requires a full-rank positive-definite covariance")
    L = q @ np.diag(np.sqrt(ev))
    expr = str(payload.get("limit_state", "")).strip()
    if not expr:
        raise InputError("limit_state expression is required; failure is defined as g(x)<=0")
    return names, mean, cov, L, expr


def _g_grad_hess_u(names, mean, L, expr, u):
    x = mean + L @ np.asarray(u, dtype=float)
    r = ad.differentiate(expr, names, x)
    g = float(r["values"][0])
    gx = np.asarray(r["jacobian"][0], dtype=float)
    hx = np.asarray(r["hessians"][0], dtype=float)
    gu = L.T @ gx
    hu = L.T @ hx @ L
    return g, gu, hu, x


def form(payload: Mapping[str, Any]) -> dict[str, Any]:
    names, mean, cov, L, expr = _normal_model(payload)
    g0, grad0, _, _ = _g_grad_hess_u(names, mean, L, expr, np.zeros(len(names),dtype=float))
    if g0 <= 0 and not bool(payload.get("allow_mean_in_failure", False)):
        raise InputError("mean point lies in the failure domain; default FORM orientation assumes a safe mean (g(mean)>0)")
    # SLSQP must see a dimensionless, order-one equality constraint. Without this
    # normalization, a physically identical problem expressed in very small units
    # can fail with a singular LSQ subproblem even though the U-space geometry is unchanged.
    constraint_scale=max(abs(float(g0)),float(np.linalg.norm(grad0)),np.finfo(float).tiny)

    def obj(u):
        return 0.5 * float(np.dot(u, u))

    def jac_obj(u):
        return np.asarray(u, dtype=float)

    def con(u):
        return _g_grad_hess_u(names, mean, L, expr, u)[0]/constraint_scale

    def jac_con(u):
        return _g_grad_hess_u(names, mean, L, expr, u)[1]/constraint_scale

    x0 = np.zeros(len(names), dtype=float)
    res = optimize.minimize(obj, x0, jac=jac_obj, constraints={"type": "eq", "fun": con, "jac": jac_con}, method="SLSQP",
                            options={"ftol": float(payload.get("optimizer_ftol", 1e-12)), "maxiter": int(payload.get("optimizer_maxiter", 1000))})
    if not res.success:
        raise InputError(f"FORM design-point optimization failed: {res.message}")
    ustar = np.asarray(res.x, dtype=float)
    gstar, grad_u, hess_u, xstar = _g_grad_hess_u(names, mean, L, expr, ustar)
    beta = float(np.linalg.norm(ustar))
    if beta == 0 or np.linalg.norm(grad_u) == 0:
        raise InputError("degenerate FORM design point/gradient")
    pf = float(stats.norm.cdf(-beta)) if g0 > 0 else float(stats.norm.cdf(beta))
    importance = (ustar / beta) ** 2
    return {
        "engine_version": ENGINE_VERSION,
        "method": "FORM_Hasofer_Lind_standard_normal_space",
        "status": "PASS",
        "failure_definition": "g(x)<=0",
        "mean_limit_state": float(g0),
        "beta": beta,
        "failure_probability_FORM": pf,
        "design_point_u": ustar.tolist(),
        "design_point_x": xstar.tolist(),
        "limit_state_at_design_point": float(gstar),
        "gradient_u": grad_u.tolist(),
        "importance_factors_u_squared": importance.tolist(),
        "optimizer": {"iterations": int(res.nit), "constraint_abs": abs(float(gstar)), "constraint_normalization_scale": constraint_scale},
        "claim_boundary": "FORM is a first-order tangent approximation except when the transformed limit state is linear, where it is exact.",
        "_hessian_u": hess_u.tolist(),
    }


def sorm(payload: Mapping[str, Any]) -> dict[str, Any]:
    base = form(payload)
    names, mean, cov, L, expr = _normal_model(payload)
    u = np.asarray(base["design_point_u"], dtype=float)
    _, grad, hess, _ = _g_grad_hess_u(names, mean, L, expr, u)
    beta = float(base["beta"])
    nrm = float(np.linalg.norm(grad))
    alpha = grad / nrm
    # Orthonormal tangent basis from null space of alpha^T.
    _, _, vh = np.linalg.svd(alpha.reshape(1, -1), full_matrices=True)
    tangent = vh[1:].T
    if tangent.shape[1] == 0:
        curv = np.array([], dtype=float)
    else:
        # Surface curvature with orientation chosen toward increasing g (safe domain).
        B = tangent.T @ hess @ tangent / nrm
        curv = np.linalg.eigvalsh((B + B.T) / 2.0)
    terms = 1.0 + beta * curv
    if np.any(terms <= 0):
        return {**{k: v for k, v in base.items() if not k.startswith("_")}, "method": "SORM_Breitung", "status": "BLOCKED",
                "principal_curvatures": curv.tolist(), "message": "Breitung correction has non-positive curvature factor; SORM approximation not promoted."}
    correction = float(np.prod(terms ** -0.5)) if terms.size else 1.0
    pf = float(base["failure_probability_FORM"] * correction)
    return {**{k: v for k, v in base.items() if not k.startswith("_")}, "method": "SORM_Breitung", "status": "PASS",
            "principal_curvatures": curv.tolist(), "Breitung_correction": correction, "failure_probability_SORM_Breitung": pf,
            "claim_boundary": "SORM uses the local quadratic curvature at the FORM design point; validity still depends on a dominant design point and a suitable smooth limit-state surface."}


def importance_sampling(payload: Mapping[str, Any]) -> dict[str, Any]:
    base = form(payload)
    names, mean, cov, L, expr = _normal_model(payload)
    n = int(payload.get("n", 100_000)); seed = int(payload.get("seed", 1729))
    if n < 1000 or n > MAX_SAMPLES:
        raise InputError(f"n must lie in [1000,{MAX_SAMPLES}]")
    shift = np.asarray(base["design_point_u"], dtype=float)
    rng = np.random.Generator(np.random.PCG64(seed))
    v = rng.normal(size=(n, len(names)))
    u = v + shift
    x = mean[None, :] + u @ L.T
    node = joint._compile(expr)
    with np.errstate(all="ignore"):
        g = np.asarray(joint._eval(node, {nm: x[:, i] for i, nm in enumerate(names)}), dtype=float)
    if g.ndim == 0: g = np.full(n, float(g))
    if g.shape != (n,) or not np.isfinite(g).all():
        raise InputError("limit-state evaluation failed during importance sampling")
    logw = -0.5 * np.sum(u * u, axis=1) + 0.5 * np.sum(v * v, axis=1)
    vals = np.where(g <= 0, np.exp(np.clip(logw, -745, 709)), 0.0)
    phat = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / math.sqrt(n))
    cv = None if phat == 0 else se / phat
    z = float(stats.norm.ppf(0.975))
    return {"engine_version": ENGINE_VERSION, "method": "importance_sampling_design_point_shift", "status": "PASS" if phat > 0 else "NOT_RESOLVED",
            "n": n, "seed": seed, "proposal_shift_u": shift.tolist(), "failure_count_under_proposal": int(np.sum(g <= 0)),
            "failure_probability": phat, "standard_error": se, "coefficient_of_variation": cv,
            "normal_approx_95_interval": [max(0.0, phat-z*se), min(1.0, phat+z*se)],
            "FORM_reference": base["failure_probability_FORM"], "claim_boundary": "The reported standard error is for the iid importance-sampling estimator under the explicit shifted-normal proposal."}


def _subset_once(payload: Mapping[str, Any], seed: int) -> dict[str, Any]:
    names, mean, cov, L, expr = _normal_model(payload)
    n = int(payload.get("n", 4000)); p0 = float(payload.get("p0", 0.1)); max_levels = int(payload.get("max_levels", 12)); proposal = float(payload.get("proposal_scale", 0.8))
    if n < 1000 or n > 200_000 or not 0.02 <= p0 <= 0.3 or max_levels < 1 or max_levels > 30 or proposal <= 0:
        raise InputError("invalid subset-simulation settings")
    nseed = max(1, int(round(p0*n)))
    chain_len = int(math.ceil(n / nseed))
    rng = np.random.Generator(np.random.PCG64(seed))
    node = joint._compile(expr)

    def geval(U):
        X = mean[None,:] + U @ L.T
        with np.errstate(all="ignore"):
            gg = np.asarray(joint._eval(node, {nm:X[:,i] for i,nm in enumerate(names)}), dtype=float)
        if gg.ndim == 0: gg=np.full(U.shape[0],float(gg))
        if gg.shape != (U.shape[0],) or not np.isfinite(gg).all(): raise InputError("non-finite subset-simulation limit state")
        return gg

    U = rng.normal(size=(n, len(names))); G = geval(U); thresholds=[]; level=0; accepts=[]
    while True:
        failure_fraction = float(np.mean(G <= 0))
        order = np.argsort(G)
        kth = min(n-1, nseed-1)
        b = float(G[order[kth]])
        if b <= 0 or level >= max_levels:
            pf = (p0 ** level) * failure_fraction
            status = "PASS" if b <= 0 and failure_fraction > 0 else "NOT_CONVERGED"
            return {"status":status,"failure_probability":float(pf),"levels":level,"thresholds":thresholds,
                    "final_failure_fraction":failure_fraction,"acceptance_rates":accepts,"n_per_level":n,"p0":p0,"seed":seed}
        thresholds.append(b)
        seeds = U[order[:nseed]].copy()
        out=[]; accepted=0; attempted=0
        for s in seeds:
            cur=s.copy(); curg=float(geval(cur.reshape(1,-1))[0]); out.append(cur.copy())
            for _ in range(chain_len-1):
                prop=cur + proposal*rng.normal(size=len(names)); pg=float(geval(prop.reshape(1,-1))[0]); attempted+=1
                if pg <= b:
                    logr=-0.5*(float(prop@prop)-float(cur@cur))
                    if math.log(rng.random()) < min(0.0,logr): cur=prop;curg=pg;accepted+=1
                out.append(cur.copy())
                if len(out)>=n: break
            if len(out)>=n: break
        U=np.asarray(out[:n],dtype=float);G=geval(U);accepts.append(None if attempted==0 else accepted/attempted);level+=1


def subset_simulation(payload: Mapping[str, Any]) -> dict[str, Any]:
    reps=int(payload.get("replicates",3));seed=int(payload.get("seed",1729))
    if reps<1 or reps>20:raise InputError("replicates must lie in [1,20]")
    runs=[_subset_once(payload,seed+i*1000003) for i in range(reps)]
    good=[r for r in runs if r['status']=='PASS']
    if not good:
        return {"engine_version":ENGINE_VERSION,"method":"subset_simulation","status":"NOT_CONVERGED","replicates":runs}
    vals=np.array([r['failure_probability'] for r in good],dtype=float)
    return {"engine_version":ENGINE_VERSION,"method":"subset_simulation","status":"PASS" if len(good)==reps else "PROVISIONAL",
            "failure_probability_mean":float(vals.mean()),"failure_probability_sd_across_replicates":float(vals.std(ddof=1)) if len(vals)>1 else None,
            "successful_replicates":len(good),"requested_replicates":reps,"replicates":runs,
            "claim_boundary":"Uncertainty is estimated from independent subset-simulation replicates; within-chain dependence is not treated as iid."}


def run_all(payload: Mapping[str, Any]) -> dict[str, Any]:
    f=form(payload); s=sorm(payload); imp=importance_sampling(payload); sub=subset_simulation(payload)
    return {"engine_version":ENGINE_VERSION,"status":"PASS" if f['status']=='PASS' and imp['status']=='PASS' and sub['status'] in {'PASS','PROVISIONAL'} else 'PROVISIONAL',
            "FORM":f,"SORM":s,"importance_sampling":imp,"subset_simulation":sub}
