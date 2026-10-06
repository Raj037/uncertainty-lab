#!/usr/bin/env python3
"""Posterior / MCMC diagnostics for Uncertainty Lab v0.6.

Implements rank-normalized split-Rhat (including folded Rhat), rank-based bulk ESS,
tail ESS from 5%/95% indicators, and mean MCSE for equal-weight multiple-chain MCMC.
Weighted posterior samples are handled separately with importance-weight diagnostics;
weights are never treated as evidence of MCMC chain mixing.

The implementation is intentionally diagnostic, not a claim of sampler correctness or
posterior-model validity.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from scipy import stats

ENGINE_VERSION = "0.6.0"
MAX_TOTAL_VALUES = 5_000_000
MIN_POST_WARMUP = 40


class InputError(ValueError):
    pass


def _split_chains(x: np.ndarray) -> np.ndarray:
    m, n = x.shape
    h = n // 2
    if h < 2:
        raise InputError("too few post-warmup draws to split chains")
    return np.concatenate([x[:, :h], x[:, n-h:]], axis=0)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    flat = np.asarray(x, dtype=float).reshape(-1)
    ranks = stats.rankdata(flat, method="average")
    # Blom offset used for stable finite rank-normal scores.
    p = (ranks - 3.0 / 8.0) / (len(flat) + 1.0 / 4.0)
    z = stats.norm.ppf(p)
    return z.reshape(np.asarray(x).shape)


def _basic_rhat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m, n = x.shape
    if m < 2 or n < 2:
        return math.inf
    vars_ = np.var(x, axis=1, ddof=1)
    means = np.mean(x, axis=1)
    W = float(np.mean(vars_))
    B = float(n * np.var(means, ddof=1))
    if W == 0.0:
        return 1.0 if B == 0.0 else math.inf
    var_plus = (n - 1.0) / n * W + B / n
    return float(math.sqrt(max(var_plus / W, 0.0)))


def _rank_split_rhat(x: np.ndarray) -> dict[str, float]:
    split = _split_chains(np.asarray(x, dtype=float))
    z = _rank_normalize(split)
    rank_rhat = _basic_rhat(z)
    med = float(np.median(split))
    folded = _rank_normalize(np.abs(split - med))
    folded_rhat = _basic_rhat(folded)
    return {
        "rank_normalized_split_rhat": rank_rhat,
        "folded_rank_normalized_split_rhat": folded_rhat,
        "rhat": max(rank_rhat, folded_rhat),
    }


def _autocov_fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    y = x - np.mean(x)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(y, n=size)
    ac = np.fft.irfft(f * np.conjugate(f), n=size)[:n]
    return np.asarray(ac / n, dtype=float)


def _ess_from_split(x: np.ndarray) -> float:
    """Geyer initial-positive/monotone sequence ESS on split chains."""
    x = np.asarray(x, dtype=float)
    m, n = x.shape
    if m < 2 or n < 3:
        return float(m * n)
    chain_var = np.var(x, axis=1, ddof=1)
    W = float(np.mean(chain_var))
    chain_mean = np.mean(x, axis=1)
    B = float(n * np.var(chain_mean, ddof=1))
    var_plus = (n - 1.0) / n * W + B / n
    total = float(m * n)
    if not math.isfinite(var_plus) or var_plus <= 0.0:
        return total if W == 0.0 and B == 0.0 else 1.0

    acovs = np.vstack([_autocov_fft(row) for row in x])
    rho = [1.0]
    max_lag = n - 1
    for t in range(1, max_lag + 1):
        mean_acov = float(np.mean(acovs[:, t]))
        r = 1.0 - (W - mean_acov) / var_plus
        # Numerical correlation estimates may slightly exceed one; retain negative
        # evidence but cap impossible positive roundoff.
        rho.append(min(1.0, r))

    pairs: list[float] = []
    k = 0
    while 2 * k + 1 < len(rho):
        p = rho[2 * k] + rho[2 * k + 1]
        if p < 0.0:
            break
        if pairs and p > pairs[-1]:
            p = pairs[-1]
        pairs.append(p)
        k += 1
    if not pairs:
        tau = 1.0
    else:
        tau = -1.0 + 2.0 * float(np.sum(pairs))
        tau = max(tau, 1.0 / total)
    ess = total / tau
    return float(min(total, max(1.0, ess)))


def _bulk_tail_ess(x: np.ndarray) -> dict[str, float]:
    split = _split_chains(np.asarray(x, dtype=float))
    z = _rank_normalize(split)
    bulk = _ess_from_split(z)
    flat = split.reshape(-1)
    q05, q95 = np.quantile(flat, [0.05, 0.95])
    low = (split <= q05).astype(float)
    high = (split >= q95).astype(float)
    ess_low = _ess_from_split(low)
    ess_high = _ess_from_split(high)
    return {
        "bulk_ess": bulk,
        "tail_ess": min(ess_low, ess_high),
        "lower_tail_ess": ess_low,
        "upper_tail_ess": ess_high,
    }


def diagnose_mcmc(payload: Mapping[str, Any]) -> dict[str, Any]:
    chains = np.asarray(payload.get("chains"), dtype=float)
    if chains.ndim == 2:
        chains = chains[:, :, None]
    if chains.ndim != 3:
        raise InputError("chains must have shape [chains, draws, parameters]")
    m, n0, d = chains.shape
    if m < 2:
        raise InputError("MCMC convergence diagnostics require at least two chains")
    if d < 1 or m * n0 * d > MAX_TOTAL_VALUES:
        raise InputError("MCMC diagnostic input exceeds resource limits")
    if not np.isfinite(chains).all():
        raise InputError("chains contain non-finite values")
    warmup = int(payload.get("warmup", 0))
    if warmup < 0 or warmup >= n0:
        raise InputError("warmup must be in [0, draws-1]")
    chains = chains[:, warmup:, :]
    n = chains.shape[1]
    if n < MIN_POST_WARMUP:
        raise InputError(f"need at least {MIN_POST_WARMUP} post-warmup draws per chain")
    names = list(payload.get("parameter_names") or [f"theta{i}" for i in range(d)])
    if len(names) != d or len(set(names)) != d:
        raise InputError("parameter_names must be unique and match parameter dimension")

    max_rhat = float(payload.get("max_rhat", 1.01))
    min_bulk = float(payload.get("min_bulk_ess", 400.0))
    min_tail = float(payload.get("min_tail_ess", 400.0))
    if not (math.isfinite(max_rhat) and max_rhat >= 1.0 and math.isfinite(min_bulk) and min_bulk > 0 and math.isfinite(min_tail) and min_tail > 0):
        raise InputError("invalid diagnostic thresholds")

    rows = []
    all_ok = True
    for j, name in enumerate(names):
        x = chains[:, :, j]
        rh = _rank_split_rhat(x)
        es = _bulk_tail_ess(x)
        pooled_sd = float(np.std(x, ddof=1))
        mcse_mean = pooled_sd / math.sqrt(max(es["bulk_ess"], 1.0))
        ok = rh["rhat"] <= max_rhat and es["bulk_ess"] >= min_bulk and es["tail_ess"] >= min_tail
        all_ok = all_ok and ok
        rows.append({
            "parameter": name,
            **rh,
            **es,
            "pooled_mean": float(np.mean(x)),
            "pooled_sd": pooled_sd,
            "mcse_mean": mcse_mean,
            "passes_thresholds": bool(ok),
        })

    return {
        "engine_version": ENGINE_VERSION,
        "kind": "rank_normalized_split_mcmc_diagnostics",
        "status": "QUALIFIED" if all_ok else "NOT_QUALIFIED",
        "chains": int(m),
        "draws_per_chain_input": int(n0),
        "warmup_discarded": warmup,
        "post_warmup_draws_per_chain": int(n),
        "parameter_names": names,
        "thresholds": {"max_rhat": max_rhat, "min_bulk_ess": min_bulk, "min_tail_ess": min_tail},
        "parameters": rows,
        "claim_boundary": (
            "These diagnostics assess finite-chain mixing/effective sample size for the supplied draws. "
            "They do not prove convergence to the intended posterior, model correctness, or absence of multimodality not visited by any chain."
        ),
    }


def diagnose_weighted(payload: Mapping[str, Any]) -> dict[str, Any]:
    samples = np.asarray(payload.get("samples"), dtype=float)
    weights = np.asarray(payload.get("weights"), dtype=float)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or weights.shape != (samples.shape[0],):
        raise InputError("samples must be N x D and weights length N")
    n, d = samples.shape
    if n < 20 or n * d > MAX_TOTAL_VALUES or not np.isfinite(samples).all() or not np.isfinite(weights).all():
        raise InputError("invalid/oversized weighted posterior input")
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise InputError("weights must be nonnegative with positive sum")
    w = weights / np.sum(weights)
    ess = float(1.0 / np.sum(w * w))
    positive = w[w > 0.0]
    entropy_ess = float(math.exp(-np.sum(positive * np.log(positive))))
    max_weight = float(np.max(w))
    min_ess = float(payload.get("min_weight_ess", 400.0))
    max_weight_limit = float(payload.get("max_normalized_weight", 0.05))
    if min_ess <= 0 or not 0 < max_weight_limit <= 1:
        raise InputError("invalid weight-quality thresholds")
    quality = ess >= min_ess and max_weight <= max_weight_limit
    names = list(payload.get("parameter_names") or [f"theta{i}" for i in range(d)])
    if len(names) != d or len(set(names)) != d:
        raise InputError("parameter_names must match sample columns")
    mean = np.sum(samples * w[:, None], axis=0)
    var = np.sum(w[:, None] * (samples - mean) ** 2, axis=0)
    return {
        "engine_version": ENGINE_VERSION,
        "kind": "weighted_posterior_quality",
        "status": "WEIGHT_QUALITY_PASS" if quality else "WEIGHT_QUALITY_FAIL",
        "sample_count": int(n),
        "parameter_names": names,
        "weight_ess": ess,
        "entropy_ess": entropy_ess,
        "max_normalized_weight": max_weight,
        "weighted_mean": mean.tolist(),
        "weighted_sd": np.sqrt(np.clip(var, 0.0, None)).tolist(),
        "thresholds": {"min_weight_ess": min_ess, "max_normalized_weight": max_weight_limit},
        "mcmc_convergence": "NOT_ASSESSED",
        "claim_boundary": (
            "Weight diagnostics quantify importance/particle weight degeneracy only. They are not R-hat or chain-mixing evidence and do not establish MCMC convergence."
        ),
    }


def analyze(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "chains" in payload:
        if "weights" in payload:
            raise InputError("weighted-chain R-hat semantics are not supported; diagnose chains and weights separately")
        return diagnose_mcmc(payload)
    if "samples" in payload and "weights" in payload:
        return diagnose_weighted(payload)
    raise InputError("supply chains for MCMC diagnostics or samples+weights for weighted posterior diagnostics")
