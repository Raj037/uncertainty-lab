# v0.9 capability contract

## Constrained transformed-space NUTS
`constrained_nuts_engine.py` samples in explicit unconstrained latent coordinates and maps to real-normal, positive lognormal, bounded logit-normal, or simplex logistic-normal physical parameters. The qualified route adapts a diagonal Euclidean metric during warm-up and retains the inherited rank-normalized R-hat, bulk/tail ESS and post-warmup divergence gates. A constrained transform is part of the probabilistic model; clipping/reflection at boundaries is not used.

## Epistemic / p-box uncertainty
`pbox_engine.py` separates aleatory variability within each candidate distribution from epistemic uncertainty about distribution parameters. Scalar CDF envelopes are optimized over declared normal/lognormal/uniform parameter boxes. Nonlinear multivariate propagation reports an envelope over the explicitly evaluated parameter grid unless a global extremum has been proven separately.

## Time-dependent reliability
`time_reliability_engine.py` supports constant-drift Brownian continuous first passage with Brownian-bridge correction, grid-monitored Ornstein-Uhlenbeck first passage, and discrete AR(1) first passage. OU/AR(1) grid probabilities are not relabeled as continuous-time crossing probabilities.

## General Boolean system reliability
`boolean_system_engine.py` evaluates arbitrary bounded Boolean system expressions (`and`, `or`, `not`) over component limit-state failures on the same explicit joint input samples. This preserves common-cause dependence. Minimal cut sets are derived exactly only for the bounded supported component count.

## Multi-fidelity UQ
`multifidelity_engine.py` provides an independent-sample control-variate estimator and an autoregressive multi-fidelity GP (`y_H = rho y_L + delta`). High-fidelity held-out validation is authoritative for surrogate qualification. The GP predictive-variance combination states its low/discrepancy posterior-independence approximation explicitly.

## Global / nonlinear experiment design
`global_design_engine.py` provides exhaustive small-batch optimization for the established linear-Gaussian design and nested-Monte-Carlo expected information gain for nonlinear Gaussian-noise candidate measurements represented over explicit parameter particles. Small nonlinear candidate batches can be exhaustively enumerated. EIG estimator uncertainty is preserved.
