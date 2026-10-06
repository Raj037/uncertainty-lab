#!/usr/bin/env python3
"""Integrated deterministic release gate for Uncertainty Lab v0.6.0."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

import v05_release_gate as v05_gate
import gum5_full_benchmarks as gum16
import adaptive_mc_engine as amc
import precision_engine as prec
import dependent_attribution_engine as dep
import posterior_diagnostics as post

ENGINE_VERSION = "0.6.0"


def _row(name: str, passed: bool, evidence: Any = None) -> dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "evidence": evidence}


def run(seed: int = 20261005) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    # 1. Parent v0.5 cross-capability gate must remain green.
    parent = v05_gate.run(seed)
    rows.append(_row("parent_v05_release_gate", parent["status"] == "PASS", {"status": parent["status"], "rows": parent["rows"]}))

    # 2. Every official GUM-5 example is represented by a numerical hard gate.
    g = gum16.run_all()
    rows.append(_row("gum5_all_16_examples", g["status"] == "PASS" and len(g["cases"]) == 16,
                     {"status": g["status"], "case_statuses": [x["status"] for x in g["cases"]], "claim_boundary": g["claim_boundary"]}))

    # 3. Adaptive MC must both converge when tolerances are supportable and refuse
    #    to claim convergence when a hard sample ceiling prevents the target precision.
    base = {
        "variables": ["x"],
        "joint_model": {"kind": "multivariate_normal", "mean": [0.0], "covariance": [[1.0]]},
        "expression": "x",
        "seed": seed,
        "batch_size": 10000,
        "min_samples": 40000,
        "coverage_probability": .95,
    }
    ok = amc.propagate({**base, "max_samples": 160000, "mc_abs_tolerance": .025, "mc_rel_tolerance": .01})
    strict = amc.propagate({**base, "max_samples": 40000, "mc_abs_tolerance": 1e-8, "mc_rel_tolerance": 0.0})
    mc_pass = ok["status"] == "CONVERGED" and strict["status"] == "NOT_CONVERGED"
    rows.append(_row("adaptive_mc_convergence_and_stop", mc_pass,
                     {"converged_status": ok["status"], "converged_n": ok["n_generated"],
                      "ceiling_status": strict["status"], "ceiling_n": strict["n_generated"]}))

    # 4. Precision escalation: an ordinary case should agree, a cancellation case
    #    must expose a float/high-precision disagreement, and interval evaluation
    #    must provide an enclosure without mislabelling it as probabilistic coverage.
    ordinary = prec.high_precision_propagate({"variables": ["x"], "values": [.3], "covariance": [[.01]], "expression": "exp(x)", "dps": 80})
    cancellation = prec.high_precision_propagate({"variables": ["x"], "values": [1e-30], "covariance": [[1e-62]], "expression": "(1+x)-1", "dps": 100})
    iv = prec.interval_evaluate({"variables": ["x"], "expression": "exp(x)", "interval_bounds": {"x": [0, 1]}})
    lo, hi = iv["intervals"][0]
    precision_pass = ordinary["status"] == "PASS" and cancellation["status"] == "FLOAT_DISAGREEMENT" and lo <= 1.0 and hi >= math.e
    rows.append(_row("precision_escalation", precision_pass,
                     {"ordinary": ordinary["status"], "cancellation": cancellation["status"], "exp_interval": [lo, hi]}))

    # 5. Correlated non-Gaussian Shapley effects: symmetric t-copula + equal beta
    #    marginals + symmetric additive output should yield approximately equal effects.
    shap = dep.shapley_empirical({
        "variables": ["a", "b"],
        "joint_model": {"kind": "t_copula", "df": 5, "copula_correlation": [[1, .65], [.65, 1]],
                        "marginals": [{"kind": "beta", "a": 2, "b": 5}, {"kind": "beta", "a": 2, "b": 5}]},
        "n": 10000, "seed": seed, "attribution_seed": seed + 17,
        "expression": "a+b", "replicates": 4, "k_neighbors": 50,
        "max_effect_standard_error": .05,
    })
    eff = np.asarray(shap["effects_mean"], dtype=float)
    shap_pass = shap["quality"] == "STABLE" and abs(float(eff.sum()) - 1.0) < 1e-10 and np.all(np.abs(eff - .5) < .09)
    rows.append(_row("correlated_nongaussian_shapley", shap_pass,
                     {"effects": eff.tolist(), "standard_error": shap["effects_standard_error"], "quality": shap["quality"]}))

    # 6. Posterior diagnostics must accept well-mixed chains and reject a clear
    #    chain-location contradiction on the same thresholds.
    rng = np.random.default_rng(seed)
    good = rng.normal(size=(4, 1800, 2))
    good_r = post.diagnose_mcmc({"chains": good.tolist(), "parameter_names": ["a", "b"]})
    bad = rng.normal(size=(4, 1600, 1)); bad[3, :, 0] += 1.5
    bad_r = post.diagnose_mcmc({"chains": bad.tolist(), "parameter_names": ["theta"]})
    posterior_pass = good_r["status"] == "QUALIFIED" and bad_r["status"] == "NOT_QUALIFIED" and bad_r["parameters"][0]["rhat"] > 1.01
    rows.append(_row("posterior_chain_diagnostics", posterior_pass,
                     {"good_status": good_r["status"], "good_max_rhat": max(x["rhat"] for x in good_r["parameters"]),
                      "bad_status": bad_r["status"], "bad_rhat": bad_r["parameters"][0]["rhat"]}))

    status = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    return {
        "engine_version": ENGINE_VERSION,
        "gate": "v0.6_integrated_release_gate",
        "status": status,
        "seed": int(seed),
        "rows": rows,
        "claim_boundary": (
            "This gate verifies the encoded v0.5 parent and v0.6 numerical capabilities, including at least one hard numerical gate for each of the 16 GUM-5:2026 examples. "
            "It is not formal JCGM/metrology accreditation and does not validate a user-supplied scientific model."
        ),
    }


if __name__ == "__main__":
    import json
    result = run()
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["status"] == "PASS" else 2)
