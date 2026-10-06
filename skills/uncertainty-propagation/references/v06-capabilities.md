# v0.6 capability contract

## New v0.6 capabilities

### All 16 GUM-5:2026 numerical gates
`gum5_full_benchmarks.py` records at least one independently recomputed published numerical quantity for each of the 16 official JCGM GUM-5:2026 examples. Some examples are full compact-model reproductions; others are explicitly scoped published sub-results where the complete example depends on larger source data or specialist machinery. A PASS is not formal standards certification or accreditation.

### Convergence-qualified Monte Carlo
`adaptive_mc_engine.py` uses deterministic independent PCG64 batch streams and continues until batch-based numerical standard errors for mean, standard deviation, and equal-tailed interval endpoints meet explicit tolerances. Reaching the hard sample ceiling without meeting all targets returns `NOT_CONVERGED` rather than a success claim.

### High-precision and interval numerical assurance
`precision_engine.py` re-evaluates the closed expression model and symbolic Jacobian at arbitrary precision and compares the result against the float/AD path without a physical-unit tolerance floor. It also supports explicit interval-box enclosures. Interval boxes are value enclosures, not probability/coverage intervals.

### Correlated non-Gaussian attribution
`dependent_attribution_engine.py` estimates conditional-expectation variance-game Shapley effects from empirical/posterior joint samples or any explicit joint model supported by the existing joint-distribution engine. Conditional expectations use cross-fitted k-nearest-neighbour regression. Replicate standard errors and estimated game non-monotonicity are reported; negative/noisy effects are not clipped.

### Posterior / MCMC diagnostics
`posterior_diagnostics.py` provides rank-normalized split-R-hat, folded R-hat, rank-based bulk ESS, 5%/95% tail ESS, mean MCSE, and explicit warm-up removal for equal-weight multiple-chain MCMC. Weighted posterior samples are routed separately to weight-ESS/entropy/max-weight diagnostics and never mislabelled as chain-mixing evidence.

## Preserved v0.5/v0.4 capabilities

v0.6 does not replace the qualified explicit-joint propagation, three-engine verification, structured covariance, joint coverage, time-series routes, existing Sobol/Gaussian-Shapley attribution, executable units, fit provenance, latent systematics, complex-linear propagation, or large published-covariance engines. Those source files remain unchanged and their evidence is inherited only when exact hashes match.
