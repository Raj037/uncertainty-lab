#!/usr/bin/env python3
"""Measurement-model adequacy and discrepancy diagnostics for Uncertainty Lab v0.7.

This module follows a fail-closed interpretation of JCGM GUM-6: numerical residual
checks can expose inadequacy, but non-rejection is not proof that every material
physical effect has been modeled. An explicit effect register is therefore kept
separate from statistical residual evidence.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import optimize, stats

import joint_distribution_engine as joint

ENGINE_VERSION = "0.7.0"
MAX_OBSERVATIONS = 200_000
MAX_MODELS = 64


class InputError(ValueError):
    pass


def _vec(x: Any, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size < 2 or a.size > MAX_OBSERVATIONS or not np.isfinite(a).all():
        raise InputError(f"{name} must be a finite vector with 2..{MAX_OBSERVATIONS} entries")
    return a


def _covariance(payload: Mapping[str, Any], n: int) -> np.ndarray:
    if "covariance" in payload:
        c = np.asarray(payload["covariance"], dtype=float)
        if c.shape != (n, n):
            raise InputError("covariance dimension mismatch")
        try:
            return joint.validate_covariance(c, "observation covariance")
        except Exception as exc:
            raise InputError(str(exc)) from exc
    if "standard_uncertainties" in payload:
        u = np.asarray(payload["standard_uncertainties"], dtype=float)
        if u.shape != (n,) or not np.isfinite(u).all() or np.any(u <= 0):
            raise InputError("standard_uncertainties must be finite, positive, and aligned")
        return np.diag(u * u)
    raise InputError("supply covariance or standard_uncertainties")


def _pinv_psd(c: np.ndarray) -> tuple[np.ndarray, float, int, np.ndarray, np.ndarray]:
    s = (c + c.T) / 2.0
    ev, q = np.linalg.eigh(s)
    scale = float(np.max(np.abs(ev))) if ev.size else 0.0
    tol = (1e-12 + 1e-10) * scale if scale > 0 else 0.0
    if ev.size and ev[0] < -tol:
        raise InputError("covariance is not positive semidefinite")
    keep = ev > tol
    if not np.any(keep):
        raise InputError("covariance has no positive-variance direction")
    inv = (q[:, keep] / ev[keep]) @ q[:, keep].T
    logdet_pos = float(np.sum(np.log(ev[keep])))
    return inv, logdet_pos, int(np.sum(keep)), ev, q


def _lag1(z: np.ndarray) -> float | None:
    if z.size < 3:
        return None
    c = z - np.mean(z)
    d = float(c @ c)
    return None if d == 0 else float(c[1:] @ c[:-1] / d)


def residual_diagnostics(payload: Mapping[str, Any]) -> dict[str, Any]:
    observed = _vec(payload.get("observed"), "observed")
    predicted = _vec(payload.get("predicted"), "predicted")
    if predicted.shape != observed.shape:
        raise InputError("observed/predicted length mismatch")
    n = observed.size
    c = _covariance(payload, n)
    r = observed - predicted
    inv, _, rank, ev, _ = _pinv_psd(c)
    k = int(payload.get("parameters_estimated", 0))
    if k < 0 or k >= rank:
        raise InputError("parameters_estimated must be in [0, covariance_rank)")
    dof = rank - k
    qstat = float(r @ inv @ r)
    pvalue = float(stats.chi2.sf(qstat, dof))
    diag_u = np.sqrt(np.clip(np.diag(c), 0.0, None))
    z = np.divide(r, diag_u, out=np.full_like(r, np.nan), where=diag_u > 0)
    if not np.isfinite(z).all():
        raise InputError("zero marginal variance prevents standardized-residual diagnostics")
    alpha = float(payload.get("lack_of_fit_alpha", 0.01))
    if not 0 < alpha < 1:
        raise InputError("lack_of_fit_alpha must lie in (0,1)")
    return {
        "engine_version": ENGINE_VERSION,
        "n": int(n),
        "covariance_rank": rank,
        "parameters_estimated": k,
        "chi_square": qstat,
        "degrees_of_freedom": dof,
        "chi_square_pvalue": pvalue,
        "lack_of_fit_alpha": alpha,
        "lack_of_fit_rejected": bool(pvalue < alpha),
        "residual_mean": float(np.mean(r)),
        "residual_rms": float(np.sqrt(np.mean(r * r))),
        "max_abs_standardized_residual": float(np.max(np.abs(z))),
        "lag1_standardized_residual": _lag1(z),
        "standardized_residuals": z.tolist(),
        "covariance_eigenvalue_min": float(ev[0]),
        "covariance_eigenvalue_max": float(ev[-1]),
        "interpretation": "A non-small p-value does not prove model adequacy; it only means this residual check did not reject the supplied model under the stated covariance model.",
    }


def estimate_additive_discrepancy(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Estimate an additive iid Gaussian discrepancy SD tau by marginal ML.

    This is a diagnostic model extension C -> C + tau^2 I. It is never inserted
    silently into the user's uncertainty budget.
    """
    observed = _vec(payload.get("observed"), "observed")
    predicted = _vec(payload.get("predicted"), "predicted")
    if predicted.shape != observed.shape:
        raise InputError("observed/predicted length mismatch")
    n = observed.size
    c0 = _covariance(payload, n)
    r = observed - predicted
    base_scale = max(float(np.sqrt(np.mean(np.diag(c0)))), float(np.sqrt(np.mean(r * r))), np.finfo(float).tiny)
    upper = float(payload.get("max_discrepancy_sd", 100.0 * base_scale))
    if not math.isfinite(upper) or upper <= 0:
        raise InputError("max_discrepancy_sd must be finite and positive")

    def nll(tau: float) -> float:
        c = c0 + (float(tau) ** 2) * np.eye(n)
        inv, logdet, rank, _, _ = _pinv_psd(c)
        if rank != n:
            return math.inf
        return 0.5 * (n * math.log(2 * math.pi) + logdet + float(r @ inv @ r))

    zero = nll(0.0)
    opt = optimize.minimize_scalar(nll, bounds=(0.0, upper), method="bounded", options={"xatol": max(1e-14, upper * 1e-10)})
    if not opt.success or not math.isfinite(float(opt.fun)):
        raise InputError("additive-discrepancy optimization failed")
    tau = float(opt.x)
    gain = float(zero - opt.fun)
    median_u = float(np.median(np.sqrt(np.diag(c0))))
    return {
        "engine_version": ENGINE_VERSION,
        "model": "additive_iid_gaussian_discrepancy",
        "tau_hat": tau,
        "tau_to_median_reported_u": None if median_u == 0 else tau / median_u,
        "negative_log_likelihood_tau0": float(zero),
        "negative_log_likelihood_hat": float(opt.fun),
        "log_likelihood_gain": gain,
        "boundary_caution": "tau=0 is a parameter-space boundary; ordinary chi-square likelihood-ratio calibration is not asserted here.",
        "claim_boundary": "Estimated discrepancy is diagnostic evidence, not an automatically admitted uncertainty component.",
    }


def compare_models(payload: Mapping[str, Any]) -> dict[str, Any]:
    observed = _vec(payload.get("observed"), "observed")
    n = observed.size
    c = _covariance(payload, n)
    inv, logdet, rank, _, _ = _pinv_psd(c)
    if rank != n:
        raise InputError("model comparison currently requires full-rank observation covariance")
    models = list(payload.get("models") or [])
    if not 2 <= len(models) <= MAX_MODELS:
        raise InputError(f"models must contain 2..{MAX_MODELS} candidates")
    rows = []
    for i, m in enumerate(models):
        if not isinstance(m, Mapping):
            raise InputError("each model must be an object")
        pred = _vec(m.get("predicted"), f"models[{i}].predicted")
        if pred.shape != observed.shape:
            raise InputError("candidate prediction length mismatch")
        k = int(m.get("parameter_count", 0))
        if k < 0 or k >= n:
            raise InputError("parameter_count must lie in [0,n)")
        r = observed - pred
        ll = -0.5 * (n * math.log(2 * math.pi) + logdet + float(r @ inv @ r))
        aic = 2 * k - 2 * ll
        aicc = None if n <= k + 1 else float(aic + (2 * k * (k + 1)) / (n - k - 1))
        bic = float(k * math.log(n) - 2 * ll)
        rows.append({"name": str(m.get("name", f"model_{i}")), "parameter_count": k, "log_likelihood": float(ll), "AIC": float(aic), "AICc": aicc, "BIC": bic})
    finite = [x for x in rows if x["AICc"] is not None]
    if finite:
        best = min(x["AICc"] for x in finite)
        weights = np.exp(-0.5 * np.array([x["AICc"] - best for x in finite]))
        weights /= weights.sum()
        for x, w in zip(finite, weights):
            x["AICc_weight"] = float(w)
    return {
        "engine_version": ENGINE_VERSION,
        "models": rows,
        "interpretation": "Information criteria compare the supplied candidate likelihood models; they do not establish that the candidate set contains the physically adequate model.",
    }


def _effect_register(effects: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    allowed = {"included", "negligible_with_evidence", "unresolved", "omitted"}
    rows = []
    unresolved = 0
    material_omitted = 0
    for i, e in enumerate(effects or []):
        if not isinstance(e, Mapping):
            raise InputError("effect register entries must be objects")
        name = str(e.get("name", "")).strip()
        status = str(e.get("status", "")).strip()
        if not name or status not in allowed:
            raise InputError(f"invalid effect register entry at index {i}")
        material = bool(e.get("material", True))
        evidence = e.get("evidence")
        if status == "negligible_with_evidence" and not (isinstance(evidence, str) and evidence.strip()):
            raise InputError("negligible_with_evidence requires a non-empty evidence field")
        if status == "unresolved" and material:
            unresolved += 1
        if status == "omitted" and material:
            material_omitted += 1
        rows.append({"name": name, "status": status, "material": material, "evidence": evidence})
    return {"effects": rows, "material_unresolved": unresolved, "material_omitted": material_omitted}


def assess(payload: Mapping[str, Any]) -> dict[str, Any]:
    diag = residual_diagnostics(payload)
    effects = _effect_register(payload.get("effects"))
    discrepancy = estimate_additive_discrepancy(payload) if bool(payload.get("estimate_discrepancy", True)) else None
    complete=bool(payload.get('effect_register_complete',False))
    justification=payload.get('effect_register_justification')
    completeness_supported=complete and isinstance(justification,str) and bool(justification.strip())
    if effects["material_omitted"] > 0 or diag["lack_of_fit_rejected"]:
        status = "PROVISIONAL"
        conclusion = "MODEL_INADEQUACY_EVIDENCE"
    elif effects["material_unresolved"] > 0:
        status = "PROVISIONAL"
        conclusion = "ADEQUACY_UNRESOLVED"
    elif not completeness_supported:
        status = "PROVISIONAL"
        conclusion = "EFFECT_REGISTER_COMPLETENESS_NOT_ESTABLISHED"
    else:
        status = "MODEL_READY"
        conclusion = "ADEQUACY_NOT_REJECTED_NOT_PROVEN"
    return {
        "engine_version": ENGINE_VERSION,
        "status": status,
        "adequacy_conclusion": conclusion,
        "residual_diagnostics": diag,
        "effect_register": {**effects, "complete_asserted": complete, "completeness_justification": justification, "completeness_supported": completeness_supported},
        "additive_discrepancy_diagnostic": discrepancy,
        "claim_boundary": "MODEL_READY here means the declared effect register is complete and the implemented numerical checks did not expose inadequacy; it is not proof that no omitted physical effect exists.",
    }
