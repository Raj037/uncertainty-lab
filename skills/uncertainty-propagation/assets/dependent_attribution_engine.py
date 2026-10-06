#!/usr/bin/env python3
"""Dependence-aware Shapley effects for arbitrary empirical/joint samples.

This v0.6 layer extends Uncertainty Lab attribution beyond multivariate-Gaussian
inputs without pretending a closed-form conditional law exists.  It estimates
v(S)=Var(E[Y|X_S]) by cross-fitted k-nearest-neighbour regression on joint
samples generated from an explicit supported joint model or supplied directly.

The result is an estimator with sampling/regression uncertainty, not an exact
analytic decomposition.  Negative effects or non-monotone estimated game values
are preserved and diagnosed rather than clipped to make a prettier budget.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

import joint_distribution_engine as joint

ENGINE_VERSION = "0.6.0"
MAX_DIM = 8
MIN_SAMPLES = 1000
MAX_SAMPLES = 200_000
MAX_REPLICATES = 20


class InputError(ValueError):
    pass


def _evaluate_output(payload: Mapping[str, Any], names: Sequence[str], x: np.ndarray) -> np.ndarray:
    if "output_samples" in payload:
        y = np.asarray(payload["output_samples"], dtype=float)
        if y.shape != (x.shape[0],):
            raise InputError("output_samples must have exactly one value per joint sample")
        if not np.isfinite(y).all():
            raise InputError("output_samples contain non-finite values")
        return y
    expression = payload.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise InputError("expression or output_samples is required")
    node = joint._compile(expression)
    env = {name: x[:, i] for i, name in enumerate(names)}
    with np.errstate(all="ignore"):
        y = np.asarray(joint._eval(node, env), dtype=float)
    if y.ndim == 0:
        y = np.full(x.shape[0], float(y))
    if y.shape != (x.shape[0],) or not np.isfinite(y).all():
        raise InputError("attribution expression produced invalid/non-finite values")
    return y


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0, ddof=1)
    # Constant conditioning coordinates carry no neighbourhood information.
    # Scale them by one rather than dropping columns so subset identity is preserved.
    sd = np.where(sd > 0.0, sd, 1.0)
    return (train - mu) / sd, (test - mu) / sd


def _conditional_game_value(
    mask: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    k_neighbors: int,
) -> float:
    d = x_train.shape[1]
    if mask == 0:
        return 0.0
    if mask == (1 << d) - 1:
        return float(np.var(y_test, ddof=1))
    cols = [i for i in range(d) if (mask >> i) & 1]
    tr, te = _standardize(x_train[:, cols], x_test[:, cols])
    tree = cKDTree(tr)
    k = min(int(k_neighbors), tr.shape[0])
    _, idx = tree.query(te, k=k, workers=1)
    if k == 1:
        pred = y_train[np.asarray(idx, dtype=int)]
    else:
        pred = np.mean(y_train[np.asarray(idx, dtype=int)], axis=1)
    return float(np.var(pred, ddof=1))


def _shapley_from_game(game: Mapping[int, float], d: int) -> np.ndarray:
    full = (1 << d) - 1
    total = float(game[full] - game[0])
    if not math.isfinite(total) or total <= 0.0:
        raise InputError("output variance is zero/non-finite; Shapley effects are undefined")
    fact = math.factorial
    phi = np.zeros(d, dtype=float)
    for i in range(d):
        bit = 1 << i
        for mask in range(1 << d):
            if mask & bit:
                continue
            s = mask.bit_count()
            weight = fact(s) * fact(d - s - 1) / fact(d)
            phi[i] += weight * (float(game[mask | bit]) - float(game[mask]))
    return phi / total


def _nonmonotone_edges(game: Mapping[int, float], d: int) -> list[dict[str, Any]]:
    full_var = max(float(game[(1 << d) - 1]), np.finfo(float).tiny)
    # Diagnose only material reversals; tiny estimator jitter is reported through
    # replicate uncertainty rather than counted as structural non-monotonicity.
    tol = 0.02 * full_var
    out: list[dict[str, Any]] = []
    for mask in range(1 << d):
        for i in range(d):
            bit = 1 << i
            if mask & bit:
                continue
            a = float(game[mask])
            b = float(game[mask | bit])
            if b + tol < a:
                out.append({"subset_mask": mask, "added_variable_index": i, "before": a, "after": b})
    return out


def _one_replicate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    train_fraction: float,
    k_neighbors: int,
) -> tuple[np.ndarray, dict[int, float], list[dict[str, Any]]]:
    n, d = x.shape
    rng = np.random.Generator(np.random.PCG64(seed))
    perm = rng.permutation(n)
    n_train = int(round(train_fraction * n))
    n_train = min(max(n_train, 2 * k_neighbors), n - max(200, k_neighbors))
    if n_train <= k_neighbors or n - n_train < 100:
        raise InputError("insufficient samples for requested train/test split and neighbour count")
    ti, vi = perm[:n_train], perm[n_train:]
    xt, yt = x[ti], y[ti]
    xv, yv = x[vi], y[vi]
    game: dict[int, float] = {}
    for mask in range(1 << d):
        game[mask] = _conditional_game_value(mask, xt, yt, xv, yv, k_neighbors)
    phi = _shapley_from_game(game, d)
    return phi, game, _nonmonotone_edges(game, d)


def shapley_empirical(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Estimate dependence-aware Shapley effects from arbitrary joint samples.

    The input distribution must be explicit. It can be provided as empirical /
    posterior samples or generated by any joint model supported by the v0.5 joint
    distribution engine (including Gaussian/t copulas with non-Gaussian marginals).
    """
    try:
        names, x, joint_meta = joint.generate_inputs(payload)
    except Exception as exc:
        if isinstance(exc, joint.InputError):
            raise InputError(str(exc)) from exc
        raise
    x = np.asarray(x, dtype=float)
    n, d = x.shape
    if d < 1 or d > MAX_DIM:
        raise InputError(f"empirical Shapley supports 1..{MAX_DIM} variables")
    if n < MIN_SAMPLES or n > MAX_SAMPLES:
        raise InputError(f"empirical Shapley requires {MIN_SAMPLES}..{MAX_SAMPLES} joint samples")
    if not np.isfinite(x).all():
        raise InputError("joint samples contain non-finite values")
    y = _evaluate_output(payload, names, x)
    if float(np.var(y, ddof=1)) <= 0.0:
        raise InputError("output variance is zero; Shapley effects undefined")

    reps = int(payload.get("replicates", 5))
    if reps < 2 or reps > MAX_REPLICATES:
        raise InputError(f"replicates must be in [2,{MAX_REPLICATES}] so estimator uncertainty is observable")
    seed = int(payload.get("attribution_seed", payload.get("seed", 1729)))
    train_fraction = float(payload.get("train_fraction", 0.65))
    if not 0.5 <= train_fraction <= 0.85:
        raise InputError("train_fraction must be in [0.5,0.85]")
    k_neighbors = int(payload.get("k_neighbors", max(10, min(80, round(math.sqrt(n) / 2)))))
    if k_neighbors < 3 or k_neighbors > 200:
        raise InputError("k_neighbors must be in [3,200]")

    effects: list[np.ndarray] = []
    variances: list[float] = []
    nonmono_counts: list[int] = []
    for r in range(reps):
        phi, game, nonmono = _one_replicate(
            x, y,
            seed=seed + r * 1_000_003,
            train_fraction=train_fraction,
            k_neighbors=k_neighbors,
        )
        effects.append(phi)
        variances.append(float(game[(1 << d) - 1]))
        nonmono_counts.append(len(nonmono))

    arr = np.vstack(effects)
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=1)
    se = sd / math.sqrt(reps)
    max_se = float(np.max(se))
    stability_limit = float(payload.get("max_effect_standard_error", 0.05))
    if not math.isfinite(stability_limit) or stability_limit <= 0.0:
        raise InputError("max_effect_standard_error must be positive and finite")
    quality = "STABLE" if max_se <= stability_limit else "UNSTABLE"

    return {
        "engine_version": ENGINE_VERSION,
        "kind": "shapley_empirical_conditional_expectation_knn",
        "status": "ESTIMATED",
        "quality": quality,
        "variable_order": names,
        "joint_model": joint_meta,
        "sample_count": int(n),
        "replicates": reps,
        "attribution_seed": seed,
        "train_fraction": train_fraction,
        "k_neighbors": k_neighbors,
        "effects_mean": mean.tolist(),
        "effects_sd_across_replicates": sd.tolist(),
        "effects_standard_error": se.tolist(),
        "effects_sum": float(mean.sum()),
        "max_effect_standard_error": max_se,
        "stability_threshold": stability_limit,
        "output_variance_mean": float(np.mean(variances)),
        "estimated_game_nonmonotone_edge_count_by_replicate": nonmono_counts,
        "claim_boundary": (
            "Effects estimate the conditional-expectation variance-game Shapley decomposition "
            "for the supplied/generated joint distribution. Conditional expectations are "
            "cross-fitted k-nearest-neighbour estimates; replicate dispersion is reported. "
            "This is not an exact analytic decomposition, and negative/noisy effects are not clipped."
        ),
    }
