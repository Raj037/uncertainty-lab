# v0.6 validation summary

Release-critical validation is recorded in `validation_receipt.json` and the machine-readable reference results beside this file.

Key gates:

- full Python test suite repeated three times on the frozen candidate;
- v0.5 parent cross-capability release gate remains PASS;
- all 16 GUM-5:2026 numerical example gates PASS;
- adaptive Monte Carlo demonstrates both a supported `CONVERGED` case and a hard-ceiling `NOT_CONVERGED` case;
- high-precision escalation detects cancellation that float evaluation loses, while ordinary cases agree;
- correlated non-Gaussian Shapley effects are stable across multiple deterministic seeds and reproduce a symmetric benchmark;
- MCMC diagnostics accept well-mixed chains and reject shifted/autocorrelated chains;
- rank-R-hat, bulk ESS and tail ESS are independently cross-checked against ArviZ 1.1.0 in the build environment;
- parent v0.5 engine/source identity is fail-closed in `v06_engine.py`.
