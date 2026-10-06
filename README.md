# Uncertainty Lab v1.0.0

Research-grade uncertainty modeling for scientific and engineering workflows.

Uncertainty Lab is designed around a fail-closed principle: a calculation is not promoted merely because a number can be produced. It keeps model assumptions, dependence structure, units, covariance validity, numerical verification, provenance, and decision semantics explicit.

## Core capabilities

- covariance-aware scalar and vector uncertainty propagation;
- executable dimensional analysis and unit validation;
- correlated non-Gaussian joint-distribution propagation;
- first/second-order nonlinear diagnostics and multi-engine derivative verification;
- large, structured, sparse-factor, and published covariance workflows;
- posterior/MCMC diagnostics, constrained NUTS/HMC calibration, and explicit model discrepancy;
- FORM/SORM, importance sampling, subset simulation, QMC, directional, time-dependent, and Boolean/system reliability;
- GP/PCE and multi-fidelity surrogate UQ with held-out qualification;
- p-box/epistemic uncertainty, structural model ensembles, and robust decisions under model-weight uncertainty;
- complex-valued linear and nonlinear uncertainty propagation;
- conformity/decision risk;
- sensitivity attribution;
- value-of-information and experimental-design analysis;
- JCGM GUM-5 numerical regression gates and reproducibility receipts.

## Scientific boundary

This software is a research-analysis tool, not an accredited metrology laboratory, regulatory authority, or formal certification system. Numerical verification does not prove that a user's physical model, source data, prior, loss function, candidate model set, or decision threshold is scientifically justified.

Unsupported or insufficiently identified assumptions remain `PROVISIONAL`, `BLOCKED`, or `SOURCE_CONFLICT` rather than being guessed.

## Privacy and networking

The bundled plugin is self-contained and has no developer-operated backend or network client. See `PRIVACY.md` for the package-level privacy statement.

## Validation

The public candidate is derived from the qualified private v1.0.0 scientific source without changing Python implementation files. `validation_receipt.json` records the public-package provenance and benchmark state; `references/public-release-benchmark-results.json` contains the exact public-candidate benchmark results.

## License and attribution

Copyright 2026 Rajat Aggarwal. Uncertainty Lab is licensed under the
[Apache License 2.0](LICENSE). Runtime dependencies are independently licensed
and are not vendored in this repository; see `THIRD_PARTY_NOTICES.md`.

References to BIPM/JCGM, OpenTURNS, UQpy, PyMC, ArviZ, OpenAI, ChatGPT, NumPy,
SciPy, SymPy, or other third-party names identify standards, software,
benchmarks, or interoperability targets only. Uncertainty Lab is not endorsed,
sponsored, certified, or accredited by those organizations or projects.

For citation metadata, see `CITATION.cff`.
