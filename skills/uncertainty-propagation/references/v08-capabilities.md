# v0.8 capability contract

## Non-Gaussian reliability

`non_gaussian_reliability_engine.py` adds two explicit probability-of-failure routes:

- independent Owen-scrambled Sobol QMC replicates for explicit Gaussian/t copulas and multivariate normal/t laws;
- directional simulation in an independent latent Gaussian space for multivariate-normal and Gaussian-copula/Nataf-style models.

Directional simulation requires a safe latent origin and a star-shaped/first-crossing failure geometry along sampled rays. The t-copula directional route is intentionally not claimed. A covariance matrix plus arbitrary non-Gaussian marginals still does not define a joint law.

## HMC / NUTS calibration

`hmc_nuts_engine.py` implements fixed-length HMC and a Hoffman-Gelman No-U-Turn Sampler with leapfrog integration, U-turn termination, dual-averaging step-size adaptation, post-warmup divergence tracking, and inherited rank-normalized split/folded R-hat plus bulk/tail ESS diagnostics.

The qualified v0.8 route samples in prior-standardized unconstrained coordinates and currently supports explicit normal priors. Bounded/positive prior transforms are not silently improvised; those models remain on validated alternative routes until transformations are qualified.

## Joint model-discrepancy inference

`discrepancy_inference_engine.py` implements linear-Gaussian joint inference for calibration parameters and an additive discrepancy scale `tau` under `tau^2 K`, with a user-supplied discrepancy kernel `K` and a half-normal prior on `tau`.

The engine integrates `tau` on a log grid and analytically conditions the parameter posterior at each grid point. It reports how much discrepancy-kernel variance lies in the parameter-effect column space. High parameter/discrepancy confounding remains `POSTERIOR_PROVISIONAL_IDENTIFIABILITY` rather than being hidden by a numerically narrow posterior.

## Active-learning reliability

`active_learning_reliability_engine.py` uses an RBF GP and the boundary acquisition quantity `U = |mu_g| / sigma_g`. Production mode returns the next candidate simulator locations to evaluate. An expression-oracle mode is provided for analytic regression tests and inexpensive models.

The reported probability interval is a GP classification interval over the fixed candidate population. It is not a proof of global surrogate validity or unmodeled model-form uncertainty.

## Multi-output surrogates

`multioutput_surrogate_engine.py` provides:

- multi-output PCE with a shared orthonormal basis and analytic cross-output covariance induced by common inputs;
- independently fitted output GPs whose common-input propagation preserves input-driven cross-output covariance.

Independent GP residual/predictive cross-output covariance is not modeled and is not claimed.

## System reliability

`system_reliability_engine.py` evaluates series, parallel, and k-out-of-n failure logic directly on common randomized-QMC joint samples. Component events therefore retain common-cause dependence; the engine does not substitute products of marginal component probabilities.

## Experimental design / value of information

`experimental_design_engine.py` ranks a finite set of candidate linear-Gaussian measurements using:

- expected information gain / log-determinant reduction;
- covariance-trace reduction;
- target-specific variance reduction;
- expected binary decision-entropy reduction for a linear target and threshold.

Costs can be used to form utility-per-cost. Greedy batches update the posterior covariance after each chosen candidate. This is an exact finite-candidate linear-Gaussian calculation, but greedy batch selection is not claimed to be globally combinatorial-optimal.

## Retained scope

v0.8.0 preserves the exact qualified v0.7.0 parent engines and all v0.6/v0.5/v0.4 routes beneath them. v0.8 does not replace prior capability merely to change a version label.
