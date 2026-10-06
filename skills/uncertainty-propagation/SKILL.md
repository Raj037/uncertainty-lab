---
name: uncertainty-propagation
description: Research-grade uncertainty modeling with structural model ensembles, robust decisions, nonlinear complex propagation, epistemic and aleatory uncertainty, model adequacy, reliability, Bayesian calibration, validated surrogates, experimental design, covariance/units, precision assurance, provenance, and fail-closed acceptance.
---

# Uncertainty Lab v1.0.0

Treat uncertainty propagation as a **measurement-model validation problem first** and a calculation second. A numerical answer is not promotable unless variable identity, uncertainty semantics, units, dependence, numerical domain, source provenance, and applicable numerical verification support it.

Never claim an engine, Monte Carlo run, benchmark, or certificate executed unless an execution tool actually ran it.


## v1.0 routing additions

Use `assets/v10_engine.py` for v1.0 operations. It binds to the exact qualified v0.9 parent hashes and delegates every unchanged older operation. New operations are: `model_ensemble`, `model_weight_robust_bounds`, `robust_decision`, `robust_experiment_design`, `nonlinear_complex_propagate`, and `v10_release_gate`.

For structural model uncertainty, preserve the decomposition between **within-model** covariance and **between-model** covariance. Explicit weights/evidences are assumptions/evidence about the declared candidate model set; they do not prove that omitted models are irrelevant. Robust decision guarantees apply only over the declared finite scenarios, losses/utilities, and feasible model-weight polytope.

For nonlinear complex propagation, represent the uncertain inputs in explicit `[Re(z1), Im(z1), ...]` order and preserve the full real/imag covariance. Report Hermitian covariance and pseudo-covariance, magnitude summaries, circular-phase concentration, invalid-domain fraction, and the local real-coordinate linearization. Do not apply ordinary linear complex propagation when nonlinearity is material.


## v0.9 routing additions

Use `assets/v09_engine.py` for v0.9 operations. It binds to the exact qualified v0.8 parent hashes and delegates unchanged older operations rather than rewriting them. New operations are: `constrained_nuts`, `pbox_cdf`, `pbox_probability`, `pbox_propagate`, `brownian_first_passage`, `ou_first_passage`, `ar1_first_passage`, `boolean_system_reliability`, `multifidelity_control_variate`, `multifidelity_gp`, `experiment_design_global_linear`, `experiment_design_nonlinear_eig`, `experiment_design_global_nonlinear_batch`, and `v09_release_gate`.

Treat aleatory and epistemic uncertainty separately. A p-box or parameter-family envelope must not be collapsed to one distribution without a user-justified selection rule. Grid-monitored time reliability must not be described as continuous first passage. Boolean system probability must use shared joint samples unless independence is explicitly the joint model. High-fidelity validation governs multi-fidelity surrogate admission. Nested-Monte-Carlo EIG is an estimate and must retain its estimator uncertainty.

## 1. Lifecycle / model status

Use the strongest status actually supported:

- **MODEL_READY** — supplied evidence supports the claimed model and calculation.
- **PROVISIONAL** — a material assumption/model choice is explicit but not established strongly enough for promotion.
- **BLOCKED** — required information is missing/invalid, dependence is undefined, domain/dimensional validity fails, or numerical verification fails.
- **SOURCE_CONFLICT** — supplied sources disagree on identity, order, semantics, covariance, or claimed equivalence.

Precedence: `SOURCE_CONFLICT > BLOCKED > PROVISIONAL > MODEL_READY`. Never weaken a validator to force a result.

## 2. Engine routing

Use the narrowest engine that preserves the required semantics.

### v0.8 reliability / modern calibration / design layer

`assets/v08_engine.py` is the preferred entry point for v0.8 operations and delegates all unchanged v0.7-and-earlier operations to their exact qualified parent engines. It fails closed unless critical parent hashes match.

New operations:

- `reliability_qmc_nongaussian` — independent scrambled-Sobol replicates for explicit Gaussian/t-copula and multivariate normal/t joint laws;
- `reliability_directional` — latent-Gaussian directional simulation for multivariate-normal and Gaussian-copula models under explicit first-crossing/star-shaped assumptions;
- `hmc_calibration` / `nuts_calibration` — prior-standardized HMC and NUTS with dual averaging, divergence tracking, and inherited R-hat/ESS diagnostics;
- `joint_discrepancy_inference` — linear-Gaussian joint parameter/discrepancy-scale inference with explicit discrepancy kernel and confounding diagnostics;
- `active_reliability_select` / `active_reliability_expression` — GP boundary acquisition for expensive reliability models;
- `multioutput_pce` / `multioutput_gp` — multi-output surrogate propagation with explicit cross-output covariance claim boundaries;
- `system_reliability` — direct shared-input series/parallel/k-out-of-n reliability;
- `experiment_design_rank` / `experiment_design_batch` — finite-candidate linear-Gaussian value-of-information ranking and greedy batch design;
- `v08_release_gate`.

For details and limitations read `references/v08-capabilities.md` and `references/v08-validation.md`.

### v0.7 model/reliability/calibration/decision layer

`assets/v07_engine.py` is the preferred entry point for v0.7 operations and delegates unchanged v0.6-and-earlier operations to their exact qualified parent engines. It fails closed unless critical parent hashes match.

New operations:

- `model_adequacy`, `residual_diagnostics`, `model_discrepancy_diagnostic`, `compare_models`;
- `FORM`, `SORM`, `reliability_importance_sampling`, `subset_simulation`, `reliability_all`;
- `gaussian_process_surrogate`, `pce_surrogate`, `propagate_gp_surrogate`, `propagate_pce_surrogate`;
- `linear_gaussian_calibration`, `mcmc_calibration`;
- `conformance_probability`, `conformity_decision`, `guard_band_design`, `global_decision_risk`;
- `v07_release_gate`.

### v0.6 unified assurance layer

`assets/v06_engine.py` is the preferred entry point for the new v0.6 operations and preserves every established v0.5 operation through the unchanged parent engine. It fails closed unless the exact qualified parent hashes match.

New operations:

- `gum5_full_gate` — numerical hard gates spanning all 16 official JCGM GUM-5:2026 examples, with per-example scope boundaries;
- `adaptive_mc` — sequential independent-batch Monte Carlo that returns `CONVERGED` only when numerical-error targets for reported summaries are met;
- `precision_verify` / `high_precision_propagate` — arbitrary-precision nominal/Jacobian/covariance re-evaluation and float-disagreement detection;
- `interval_enclosure` — explicit input-box value enclosure; never describe it as a probability or coverage interval;
- `shapley_correlated_nongaussian` — empirical conditional-expectation Shapley effects for arbitrary explicit joint samples/models, with cross-fit estimator uncertainty;
- `posterior_diagnostics` — rank-normalized split/folded R-hat, bulk/tail ESS, mean MCSE, warm-up handling, or separate weighted-sample quality diagnostics;
- `v06_release_gate` — integrated v0.5-parent + v0.6 release gate.

For operations introduced in v0.5, `v06_engine` delegates to the exact unchanged `v05_engine.py`; do not rewrite a qualified path merely to change its version label.

### Retained v0.5 numerical core

`assets/v05_engine.py` remains authoritative for:

- `research_propagate` / `triple_verify`;
- `joint_propagate`;
- `structured_covariance`;
- Gaussian/empirical joint coverage;
- explicit time-series routes;
- independent Sobol and correlated-Gaussian Shapley attribution;
- the v0.5 release gate.

### Retained qualified v0.4.1 routes

Do **not** replace established capability merely to use the newest file:

- `assets/advanced_engine.py` — executable units, latent/shared systematic construction, fit import/provenance, complex linear propagation, Gaussian second-order diagnostics;
- `assets/strict_engine.py` — mandatory v0.4.1 acceptance path for unit-aware/fit/latent research workflows, with scale-invariant symbolic-vs-five-point verification;
- `assets/large_covariance_engine.py` — published/large dense covariance validation and direct linear `J Sigma J^T`;
- `assets/reference_engine.py` — established Type-A/Type-B, coverage, empirical samples, and validated legacy non-Gaussian routes.

For unit-aware, fit-provenance, latent-source, or complex-linear tasks, use the established v0.4.1 route unless the same semantics are explicitly implemented and validated in v0.5.

## 3. Measurement-model adequacy and discrepancy

Use `assets/model_adequacy_engine.py` before propagation when observations and model predictions are available. It separates:

- covariance-aware residual chi-square evidence;
- standardized residual and lag-1 diagnostics;
- an explicit register of included, evidence-negligible, unresolved, and omitted effects;
- optional additive Gaussian discrepancy estimation as a **diagnostic**, never an automatically admitted uncertainty term;
- information-criterion comparison among an explicit candidate model set.

A non-small residual p-value means only **adequacy not rejected by that check**. It never proves that omitted physical effects do not exist. `MODEL_READY` from this layer additionally requires an explicitly declared complete effect register with a substantive completeness justification. Known material omitted effects, statistically incompatible residuals, or unresolved material effects remain `PROVISIONAL`. This follows the GUM-6 emphasis on developing and assessing the adequacy of the measurement model.

## 4. Rare-event / reliability analysis

Use `assets/reliability_engine.py` for failure events `g(x) <= 0` under an explicit full-rank multivariate-normal input model. Available routes are Hasofer-Lind FORM, Breitung SORM, design-point shifted-normal importance sampling, and replicated subset simulation.

FORM/SORM operate in dimensionless standard-normal `U` space and normalize the equality constraint so physical unit scaling cannot change optimizer conditioning. SORM blocks if its local Breitung curvature factors are invalid rather than returning a forced number. Importance-sampling standard errors apply to its iid weighted estimator. Subset-simulation uncertainty is assessed from independent replicates; within-chain states are not treated as iid.

Do not infer a non-normal isoprobabilistic transform automatically. If the declared joint input model is outside the qualified normal transformation, remain unresolved or use an explicitly validated simulation route.

## 5. Validated surrogates for expensive models

Use `assets/surrogate_engine.py` only when direct model evaluations are expensive or unavailable at propagation scale. The v0.7 routes are:

- RBF Gaussian-process regression, including predictive variance;
- orthonormal polynomial-chaos expansion for independent normal/uniform marginals.

A surrogate is not qualified from training fit. Qualification requires user-declared held-out error thresholds and must survive deterministic repeated train/validation splits; the worst validation metric governs acceptance. An unqualified surrogate is blocked from uncertainty propagation. GP predictive variance is reported separately from input-driven physical variance and is never silently relabeled as measurement uncertainty. PCE coefficient moments are valid only under the declared input distribution and orthonormal basis.

## 6. Bayesian calibration with explicit discrepancy

Use `assets/calibration_engine.py` for parameter calibration from measurements. The exact linear-Gaussian route is preferred whenever its assumptions hold. The general route uses explicit priors plus multi-chain random-walk Metropolis and then applies the inherited rank-normalized R-hat/bulk-tail ESS diagnostics.

Observation covariance and model-discrepancy covariance are separate explicit inputs. The calibration engine never estimates a hidden discrepancy term merely to improve fit. Posterior chain diagnostics assess finite-chain numerical quality, not scientific adequacy of the forward model or identifiability of parameter/discrepancy choices.

## 7. Conformity and decision risk

Use `assets/conformity_engine.py` for scalar JCGM-106-style tolerance/acceptance decisions. Keep distinct:

- tolerance interval: permissible true/property values;
- acceptance interval: measured values that trigger acceptance;
- guard band: separation between tolerance and acceptance limits;
- specific consumer risk: probability of nonconformity conditional on the supplied item measurement distribution when accepted;
- specific producer risk: probability of conformity when rejected;
- global risks: population joint false-accept/false-reject probabilities under an explicit population and measurement-error model.

Guard-band design is conditional on its supplied normal measurement-result model. The engine reports decision risk; it does not decide whether a regulatory, contractual, or safety-critical risk threshold is acceptable. Expanded acceptance outside tolerance requires explicit opt-in.

## 8. Dependence is never invented

A quantity used twice is one uncertain variable unless independent realizations are explicitly intended. Inspect for shared calibration, normalization, exposure/luminosity, target thickness, geometry, detector efficiency, background, common denominators, fit covariance, constrained fractions, batch effects, and ordered time dependence.

For multiple inputs with no covariance/correlation, a boolean `assume_independent=true` is not evidence by itself. `MODEL_READY` requires substantive independence justification.

For repeated observations, a boolean IID assertion is not evidence by itself. Use explicit acquisition/repeatability justification or one of the v0.5 time-series models. Diagnostics must never auto-select AR(1), HAC, bootstrap, or an effective sample size.

## 9. Explicit correlated non-Gaussian propagation

Use `assets/joint_distribution_engine.py` / `joint_propagate` only when the joint law is explicit.

Supported routes:

- supplied empirical/posterior joint samples;
- Gaussian copula with explicit latent copula correlation and marginals;
- t copula with explicit degrees of freedom, latent copula correlation, and marginals;
- multivariate normal with full covariance;
- multivariate t with full scale matrix and degrees of freedom.

Supported marginal transforms include normal, rectangular/uniform, triangular, lognormal, beta, gamma, Student-t, and empirical distributions.

A covariance matrix plus arbitrary non-Gaussian marginals does **not** define a unique joint law. If no supported joint model is supplied, remain **BLOCKED**. Copula correlation is labeled as latent dependence and must not be silently re-described as Pearson correlation after marginal transformation.

Report invalid-domain fraction; if any samples are invalid, summaries are conditional on valid samples and must be labeled accordingly.

## 10. Three-engine derivative verification

For ordinary differentiable real-valued v0.5 research models, numerical acceptance uses three independent implementations:

1. safe SymPy symbolic derivatives;
2. `assets/autodiff_engine.py` second-order forward automatic differentiation (`Jet2`), without SymPy or finite-difference steps;
3. independent AST five-point finite differences with scale-aware adaptive steps.

`assets/verification_engine.py` compares value, uncertainty-weighted Jacobian, propagated covariance, and symbolic-vs-AD Hessian on scale-invariant bases. Tiny physical covariances must not pass merely because their raw magnitude lies below an absolute tolerance. Any material engine disagreement is **BLOCKED**.

Agreement verifies implementation consistency, not the scientific validity of the user's measurement model.

## 11. Covariance integrity and scale invariance

For every covariance object verify exact variable/order identity, square finite shape, nonnegative variances, symmetry, PSD, rank/constraint semantics, and zero-variance rows carrying exactly zero covariance.

Symmetry and PSD/rank decisions use variance-normalized congruent bases where applicable so positive unit rescaling cannot turn an invalid covariance valid or manufacture rank loss. Never silently project to nearest PSD.

For published decimal matrices, a measured serialization asymmetry may use an explicit justified tolerance only in the qualified large-covariance path. A symmetry override must not relax PSD validation.

## 12. Structured / sparse covariance

Use `assets/structured_covariance_engine.py` when dense materialization is unnecessary or too expensive.

Accepted representations:

- dense (fully audited);
- diagonal (PSD by construction);
- diagonal plus low-rank factor and optional source covariance;
- sparse factor plus diagonal and optional source covariance (PSD by construction);
- block diagonal with each block audited;
- arbitrary sparse COO only up to the bounded dense-audit threshold.

Large arbitrary sparse covariance is not declared PSD from random probes or partial minors. For large systems require a factor/low-rank/block representation that proves PSD by construction or provide a separately qualified solver path.

## 8. Joint coverage regions

Scalar intervals do not substitute for a joint multivariate coverage statement.

- `gaussian_coverage_region` returns the chi-square Mahalanobis ellipsoid under an explicit multivariate normal model, including rank-deficient support.
- `empirical_coverage_region` calibrates a Mahalanobis radius from supplied joint samples; it is **descriptive/empirical**, not an exact future-sample confidence guarantee.
- `empirical_joint_box` calibrates a simultaneous max-standardized box from supplied samples.

Preserve the full output covariance and state the probability model/empirical interpretation.

## 14. Explicit time-series / repeatability models

Use `assets/timeseries_engine.py` only with an explicit model choice:

- `iid` — standard error of the mean under IID;
- `ar1` — finite-sample stationary AR(1), with supplied `phi` or explicitly requested estimation;
- `hac` — Newey-West/HAC with explicit bandwidth;
- `block_bootstrap` — deterministic circular block bootstrap with explicit block length, resample count, and seed;
- `balanced_random_effects` — balanced one-way random-intercept variance model.

Lag-1, drift, split-sample, or other diagnostics are evidence, not model selectors. Do not automatically repair dependence by choosing a model for the user.

## 15. Sensitivity / uncertainty attribution

Route attribution by dependence semantics:

- independent inputs: `sobol_independent`;
- correlated multivariate-Gaussian inputs: established `shapley_correlated_gaussian`;
- arbitrary empirical/posterior or explicitly generated correlated non-Gaussian inputs: v0.6 `shapley_correlated_nongaussian`.

The v0.6 non-Gaussian route estimates `Var(E[Y|X_S])` with cross-fitted k-nearest-neighbour conditional-mean regression and computes the complete Shapley game for up to the bounded supported dimension. Report replicate dispersion/standard errors and estimated non-monotone game edges. It is an estimator, not an exact analytic decomposition. Never clip negative or unstable effects to make them sum nicely; instability must remain visible.

Do not apply ordinary independent-input Sobol formulae to correlated inputs.

## 16. Nonlinearity, Monte Carlo, and numerical precision

The Gaussian research core retains first-order covariance plus Gaussian next-order Taylor mean/covariance terms. Its 0.1-sigma / 5% triggers are review heuristics, not universal significance thresholds.

For distribution propagation, use an explicit joint law. When a user needs a numerical-precision claim rather than merely a fixed-`N` simulation, use `adaptive_mc`: independent deterministic PCG64 batches continue until batch-based standard errors for mean, standard deviation, and equal-tailed interval endpoints satisfy explicit absolute/relative tolerances. Reaching the sample ceiling without satisfying every requested target is `NOT_CONVERGED`. A converged simulation validates numerical precision under the supplied joint model; it does not validate the joint model itself.

For cancellation, extreme scales, or suspicious floating-point agreement, use the v0.6 precision layer. `high_precision_propagate` recomputes the closed expression and symbolic Jacobian at arbitrary precision and compares against the float/AD path without a fixed physical-unit floor. A material mismatch is `FLOAT_DISAGREEMENT`, not something to repair silently. `interval_enclosure` returns a deterministic enclosure over supplied input bounds and must never be reported as a probability/coverage interval.

## 17. Units, fits, shared systematics, and complex quantities

Retain the qualified v0.4.1 behavior:

- executable units normalize values/covariance internally and block dimensional contradictions/unknown units;
- fit imports preserve exact parameter order and hash source files;
- latent/shared systematics construct covariance from named sources without inventing unique attribution when source cross-covariance makes it non-unique;
- complex support is linear/affine in explicit `[Re, Im]` covariance order and returns covariance plus pseudo-covariance; arbitrary nonlinear complex metrology is not claimed.

## 18. GUM-5:2026 benchmark claims

`assets/gum5_full_benchmarks.py` provides at least one independently recomputed published numerical hard gate for **each of the 16 official JCGM GUM-5:2026 examples**.

Gate scope is explicit. Some examples are full compact measurement-model reproductions; others reproduce a published submodel/sub-result when the complete example relies on larger source data, specialist software, or a broader workflow. Printed-input rounding envelopes are documented where the published table is less precise than the source calculation. Never convert a partial sub-result gate into a claim that the entire example has been implemented.

A 16/16 PASS is software regression evidence only. It is not formal GUM compliance certification, traceability, laboratory accreditation, or validation of an unrelated user model.

## 19. Posterior diagnostics, reproducibility, and release gates

For equal-weight multiple-chain posterior/MCMC draws, v0.6 provides rank-normalized split R-hat, folded rank-normalized R-hat, rank-based bulk ESS, 5%/95% tail ESS, and mean MCSE. Warm-up removal is explicit. A threshold PASS assesses the supplied finite chains; it does not prove convergence to the intended posterior or rule out unvisited modes.

Weighted posterior/importance samples use a separate route reporting weight ESS, entropy ESS, and maximum normalized weight. Weight quality is **not** R-hat or chain-mixing evidence and must not be promoted into an MCMC-convergence claim.

Serious v0.7 results include a certificate binding canonical payload/result, hashes of the v0.7 modules, runtime versions, and exact qualified v0.6 parent/dependency hashes. Parent-identity mismatch is fail closed. Established v0.6 operations retain their v0.6 certificates when delegated unchanged.

`assets/v07_release_gate.py` is the integrated v0.7 gate. It first requires the complete v0.6 gate to remain PASS, then checks model-adequacy fail-closed semantics, exact/rare-event reliability benchmarks, repeated held-out surrogate validation, calibration against an exact Gaussian posterior with explicit discrepancy, and JCGM-106-style guard-band/decision-risk calculations.

## 20. Reporting contract

Lead with:

1. **Model status**;
2. **Result** — estimate and standard uncertainty with units when applicable;
3. **Critical assumptions / dependence model**;
4. **Correlation/source or sensitivity budget**;
5. **Diagnostics** — covariance validity, domain/units, nonlinearity, numerical verification;
6. **Coverage** only when justified, with scalar vs joint distinction;
7. **Reproducibility** — method, seed/sample count, order, hashes/certificate.

Useful execution labels: `ANALYTIC_DERIVATION`, `NUMERICALLY_COMPUTED`, `MONTE_CARLO_EXECUTED`, `REFERENCE_CAPABILITY_ONLY`, `BLOCKED`.

## 21. Release-critical fail-closed checklist

Before promoting a release, require applicable evidence for:

- tiny valid and tiny indefinite covariance;
- extreme positive diagonal rescaling;
- asymmetry hidden at small raw scale;
- zero-variance cross-covariance;
- derivative roundoff cases where finite-difference values collapse;
- independent AD/symbolic/finite-difference agreement;
- non-Gaussian dependence without a joint law remaining blocked;
- source/order conflicts and resource ceilings;
- deterministic/input-sensitive certificates;
- structured covariance without false PSD claims;
- joint coverage calibration;
- explicit time-series choice without auto-selection;
- sensitivity method matching dependence assumptions;
- all 16 GUM-5 examples represented by explicit numerical gates with honest scope boundaries;
- adaptive Monte Carlo convergence and hard-ceiling refusal;
- high-precision cancellation regression and interval claim separation;
- correlated non-Gaussian Shapley estimator uncertainty/stability;
- posterior R-hat/ESS failure cases and weighted-sample semantic separation;
- preservation of unchanged qualified v0.6/v0.5/v0.4.1 engine evidence;
- model adequacy never inferred from non-rejection alone and effect-register completeness explicitly justified;
- FORM invariance under extreme physical-unit rescaling plus SORM curved-surface validation;
- surrogate rejection on deliberately underfit holdout cases and repeated-split qualification;
- Bayesian calibration recovery of a closed-form posterior and explicit discrepancy separation;
- conformity guard-band analytic benchmark and item-specific/global-risk semantic separation.

One material unsupported row keeps the overall release incomplete.

## References

Read as needed:

- `references/v07-capabilities.md`
- `references/v07-validation.md`
- `references/v07-release-gate-result.json`
- retained `references/v06-capabilities.md`
- `references/v06-validation.md`
- `references/gum5-full-16-result.json`
- `references/v06-release-gate-result.json`
- `references/v06-multiseed-result.json`
- `references/posterior-arviz-crosscheck.json`
- retained v0.5 references for the unchanged parent capabilities
- existing v0.4 references for units, fit provenance, large covariance, complex uncertainty, standards, reporting, and audit semantics.
