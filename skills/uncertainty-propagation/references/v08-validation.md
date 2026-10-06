# v0.8 validation notes

## Methodological references

- PyMC 6.3.2 documentation describes NUTS as the No-U-Turn sampler and documents target-acceptance and tree-depth controls. This is a methodological cross-reference; PyMC was not used as runtime acceptance authority for the bundled internal NUTS implementation.
- OpenTURNS 1.27 reliability documentation lists FORM/SORM, importance simulation, directional simulation, quasi-Monte Carlo and subset sampling as complementary reliability methods.
- OpenTURNS SystemFORM documentation and system-event examples describe union/intersection reliability events and direct shared-input event simulation.
- JCGM GUM-6:2020 explicitly discusses model selection/model uncertainty and propagation of uncertainty across plausible model alternatives; v0.8 retains v0.7 model-adequacy boundaries and does not infer a single “true” model from goodness-of-fit alone.

## Hard numerical validation

- A one-dimensional lognormal 5% tail is reproduced by both scrambled Sobol QMC and latent-Gaussian directional simulation.
- Multivariate-t QMC is deterministic for a fixed scramble seed and rejects missing/implicit joint laws.
- HMC and NUTS reproduce the exact linear-Gaussian posterior benchmark; post-warmup divergence count must be zero and inherited rank-R-hat/ESS gates must qualify.
- NUTS is regression-tested under a `1e-20` change of parameter units by sampling in prior-standardized coordinates.
- Joint discrepancy inference recovers a simulated iid discrepancy scale and parameter, while a rank-one discrepancy kernel aligned with the intercept is explicitly flagged as parameter/discrepancy confounding.
- Active-learning reliability targets the three-sigma boundary and its GP classifier probability matches the exact normal tail within the preregistered error envelope.
- Multi-output PCE reproduces the exact covariance `[[2,2],[2,4]]` for a two-output linear-normal benchmark.
- Series and parallel system probabilities reproduce independent-normal closed forms, while a correlated benchmark demonstrates that direct shared-input sampling differs materially from an independence product.
- Experimental-design information gain is invariant to positive parameter-unit rescaling when sensitivities are transformed consistently; target-specific design chooses the measurement aligned with the requested target.

## Stability

The complete test inventory contains **175 tests**. It passed in four deterministic partitions totaling `50 + 39 + 38 + 48 = 175` tests on three consecutive frozen-candidate passes. The integrated v0.8 release gate also passed for seeds `1`, `1729`, `20261005`, `8675309`, and `2147483647` after strengthening the NUTS release benchmark to 900 post-warmup draws and higher ESS requirements.

The first five-seed release-gate attempt preserved one contradiction: seed `2147483647` produced accurate posterior moments and zero post-warmup divergences but rank-R-hat `1.011047`, slightly above the preregistered `1.01` threshold at only 600 post-warmup draws. The threshold was not weakened. The release benchmark was strengthened to 900 post-warmup draws; all five seeds then qualified with rank-R-hat below `1.01`, adequate bulk/tail ESS, and zero post-warmup divergences.

## Build-time repairs

- NUTS/HMC sampling was changed to prior-standardized unconstrained coordinates after an adversarial tiny-parameter-unit test exposed potential scale dependence.
- Experimental-design covariance positive-definiteness was changed from a raw-unit eigenvalue floor to a variance-normalized basis after a `1e-20` parameter-scale test exposed false rejection.
- Active-learning zero predictive variance is handled as infinite acquisition certainty rather than generating floating-point overflow.

## Claim boundary

This validation is software evidence for the declared v0.8 scope. It is not formal reliability/metrology accreditation, proof that an expensive-model GP is globally valid, proof of model-discrepancy identifiability outside the supplied kernel/prior structure, proof that an internal NUTS sampler has visited every posterior mode, or proof that a greedy experimental design is globally optimal.
