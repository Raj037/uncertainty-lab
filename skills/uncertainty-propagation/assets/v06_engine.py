#!/usr/bin/env python3
"""Unified v0.6 interface for Uncertainty Lab.

v0.6 adds full GUM-5 numerical regression coverage, convergence-qualified Monte
Carlo, high-precision / interval assurance, arbitrary-joint empirical Shapley
attribution, and posterior/MCMC diagnostics while preserving v0.5 operations.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import v05_engine as v05
import gum5_full_benchmarks as gum16
import adaptive_mc_engine as adaptive_mc
import precision_engine as precision
import dependent_attribution_engine as dependent_attr
import posterior_diagnostics as posterior
import v06_release_gate as release_gate

ENGINE_VERSION = "0.6.0"
EXPECTED_PARENT_HASHES = {
    "v05_engine": "2d736d466c4dd95054dbf17316c6802cc0246e04f5e3931c6e28119dbe4eea8d",
    "joint_distribution_engine": "6a0701edc88d3780232cd5f3207b95652ebba0ffc5c3899913b0750ab6d8bc2f",
    "autodiff_engine": "b24750dd5d3bda86efd15d160ba51b52c060972c0878bce3fda992cb058b3606",
    "verification_engine": "1b183d381a045610fc923f0e55d9dcd393c2b2b76e88e0eadb14959cdc0c4623",
}


class InputError(ValueError):
    pass


def _sha_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def assert_parent_identity() -> dict[str, str]:
    modules = {
        "v05_engine": v05,
        "joint_distribution_engine": __import__("joint_distribution_engine"),
        "autodiff_engine": __import__("autodiff_engine"),
        "verification_engine": __import__("verification_engine"),
    }
    got: dict[str, str] = {}
    for name, module in modules.items():
        path = Path(module.__file__)
        actual = _sha_file(path)
        expected = EXPECTED_PARENT_HASHES[name]
        if actual != expected:
            raise InputError(f"v0.6 parent identity mismatch for {name}: {actual}, expected {expected}")
        got[name] = actual
    return got


def make_certificate(payload: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    modules = [
        sys.modules[name] for name in [
            "v06_engine", "gum5_full_benchmarks", "adaptive_mc_engine", "precision_engine",
            "dependent_attribution_engine", "posterior_diagnostics", "v06_release_gate"
        ] if name in sys.modules
    ]
    sources = {m.__name__: _sha_file(m.__file__) for m in modules if getattr(m, "__file__", None)}
    cert = {
        "certificate_version": "4",
        "engine": "uncertainty-lab-v06",
        "engine_version": ENGINE_VERSION,
        "parent_identity": assert_parent_identity(),
        "payload_sha256": _sha_value(payload),
        "result_sha256": _sha_value(result),
        "source_sha256": sources,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "sympy": __import__("sympy").__version__,
            "platform": platform.platform(),
        },
    }
    cert["certificate_sha256"] = _sha_value(cert)
    return cert


def _certify(payload: Mapping[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    if "certificate" not in result:
        base = dict(result)
        result = dict(result)
        result["certificate"] = make_certificate(payload, base)
    return result


def run_request(req: Mapping[str, Any]) -> dict[str, Any]:
    assert_parent_identity()
    op = str(req.get("operation", "research_propagate"))
    if op == "gum5_full_gate":
        out = gum16.run_all()
    elif op == "adaptive_mc":
        out = adaptive_mc.propagate(req)
    elif op == "precision_verify":
        out = precision.verify(req)
    elif op == "high_precision_propagate":
        out = precision.high_precision_propagate(req, int(req.get("dps", 80)))
    elif op == "interval_enclosure":
        out = precision.interval_evaluate(req)
    elif op == "shapley_correlated_nongaussian":
        out = dependent_attr.shapley_empirical(req)
    elif op == "posterior_diagnostics":
        out = posterior.analyze(req)
    elif op == "v06_release_gate":
        out = release_gate.run(int(req.get("seed", 20261005)))
    else:
        # Preserve all established v0.5 operations and their v0.5 certificates.
        return v05.run_request(req)
    return _certify(req, out)


def main() -> int:
    try:
        req = json.load(sys.stdin)
        print(json.dumps({"ok": True, "result": run_request(req)}, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__, "message": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
