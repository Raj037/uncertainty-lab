# v0.7 validation notes

## External methodological references

- **JCGM GUM-6:2020** — official BIPM document on developing and using measurement models; explicitly includes assessment of measurement-model adequacy.
- **JCGM 106:2012** — official BIPM guide on measurement uncertainty in conformity assessment; defines tolerance/acceptance intervals, guard bands, consumer risk and producer risk.
- **OpenTURNS reliability documentation** — independent reference descriptions for FORM/SORM, importance sampling and subset sampling.
- **OpenTURNS Gaussian-process documentation** — independent reference description of Gaussian-process surrogate modeling and predictive covariance.

These references support methodology and terminology. Their existence is not treated as independent execution of Uncertainty Lab.

## Hard numerical validation

- Model-adequacy chi-square diagnostics are invariant under output rescaling by `1e-30` and `1e30`; statistically incompatible residuals and unresolved effects remain provisional.
- FORM reproduces exact linear-normal failure probabilities and remains invariant when the same 4-sigma problem is rescaled to covariance `1e-40`.
- Breitung SORM is closer than FORM to independent numerical quadrature on a mild curved limit-state case; it blocks a strong-curvature case with an invalid Breitung factor.
- Design-point importance sampling resolves a 4-sigma Gaussian tail with the exact probability inside five reported estimator standard errors.
- Replicated subset simulation tracks a known Gaussian rare-event tail and is additionally exercised across five release-gate seeds.
- PCE exactly reproduces a quadratic model and its analytic mean/variance. A deliberately underfit degree-1 PCE is rejected by held-out validation.
- GP regression is qualified only after repeated held-out splits; propagation separates input-driven variance from GP predictive variance.
- Exact linear-Gaussian calibration matches closed-form posterior moments. Multi-chain Metropolis recovers that posterior within preregistered error bounds and passes inherited R-hat/ESS thresholds.
- Explicit discrepancy covariance increases posterior uncertainty; absent discrepancy remains exactly zero rather than being inferred silently.
- One-sided guarded acceptance reproduces the exact normal quantile rule. Global consumer/producer risks are deterministic for a fixed PCG64 seed and remain semantically separate from item-specific risk.

## Stability

The complete test suite passed **140/140** on three consecutive completed runs. The integrated v0.7 release gate passed for seeds `1`, `1729`, `20261005`, `8675309`, and `2147483647`.

## Claim boundary

The release is software verification evidence for the encoded scope. It is not formal metrology accreditation, reliability certification, proof that a user measurement model is physically complete, proof that a surrogate is valid outside its validated domain, or authorization of a user-specific conformity decision.
