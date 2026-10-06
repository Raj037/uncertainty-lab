# v0.9 validation notes

- Constrained NUTS passes positive, bounded, simplex, and 1e-20 physical-scale invariance cases with the inherited R-hat <= 1.01 gate, required ESS and zero post-warmup divergences.
- Parametric normal p-box CDF bounds reproduce the exact Phi(-1)/Phi(1) envelope for uncertain location, and degenerate parameter boxes collapse to a single CDF.
- Brownian first-passage simulation with Brownian-bridge correction reproduces the reflection-principle probability and remains invariant under 1e-20 physical rescaling.
- General Boolean system reliability reproduces independent series/parallel truth and correctly differs from false independence when components share a common input.
- Multi-fidelity control variates reduce estimated standard error on a known correlated low/high pair, and the autoregressive GP is qualified/rejected using repeated high-fidelity holdouts.
- Global linear-Gaussian batch design never scores below its greedy counterpart for the same finite batch problem, while nonlinear particle EIG ranks a strongly informative candidate above a weak one and can change under explicit cost normalization.
- The inherited v0.8 integrated release gate remains a mandatory first row of the v0.9 gate.
- Full suite: 208 tests. Frozen candidate passed the complete deterministic partitioned suite three times. The only later mutation was an estimator-aware Boolean Monte Carlo release-gate threshold; that gate was then rerun three times and the integrated gate passed five widely separated seeds.

Claim boundary: this is software verification for the encoded plugin-only scope. It is not formal reliability/metrology accreditation or proof that an epistemic parameter family contains every scientifically plausible distribution/model.
