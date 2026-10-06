# v0.7 capability contract

## Measurement-model adequacy / model discrepancy

`model_adequacy_engine.py` implements covariance-aware residual diagnostics, an explicit influence/effect register, candidate-model information-criterion comparison, and an optional additive iid Gaussian discrepancy diagnostic. A non-small residual p-value is **not** proof of adequacy. `MODEL_READY` additionally requires an explicitly declared complete effect register with a substantive completeness justification. Estimated discrepancy is diagnostic evidence and is never admitted automatically into the uncertainty budget.

Methodological basis: JCGM GUM-6:2020, *Developing and using measurement models*, including its explicit treatment of assessing measurement-model adequacy.

## Rare-event reliability

`reliability_engine.py` supports failure events `g(x)<=0` for an explicitly supplied full-rank multivariate-normal input model:

- Hasofer-Lind FORM;
- Breitung SORM with fail-closed curvature checks;
- design-point shifted-normal importance sampling with iid estimator SE;
- replicated subset simulation with between-run stability evidence.

The FORM equality constraint is normalized in standard-normal space so physical unit scaling cannot change optimizer conditioning. Non-normal isoprobabilistic transformations are not inferred automatically.

## Validated surrogates

`surrogate_engine.py` supports scalar-output RBF Gaussian-process regression and orthonormal polynomial-chaos expansion (independent normal/uniform inputs). Qualification requires user-declared thresholds and deterministic repeated held-out splits; the worst validation metrics govern. An unqualified surrogate is blocked from propagation. GP predictive variance and input-driven variance are reported separately.

## Bayesian calibration

`calibration_engine.py` provides:

- an exact linear-Gaussian posterior;
- a safe-expression random-walk Metropolis route with explicit priors, observation covariance, and fixed explicit model-discrepancy covariance;
- inherited rank-normalized R-hat, bulk/tail ESS and MCSE chain qualification.

The engine never estimates hidden discrepancy merely to improve fit. Posterior chain quality does not prove the forward model is scientifically adequate.

## Conformity / decision risk

`conformity_engine.py` implements scalar JCGM-106-style tolerance and acceptance intervals, normal or empirical item conformance probability, guarded acceptance, specific consumer/producer risk, and global population false-accept/false-reject risk under explicit population and measurement-error distributions.

The engine reports risk. It does not determine whether a regulatory, safety, contractual, or business risk threshold is acceptable.
