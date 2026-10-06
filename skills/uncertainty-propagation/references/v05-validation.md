# v0.5 validation policy

A release is promotable only if all release-critical rows have direct evidence on the exact candidate snapshot.

Required candidate checks:

- complete pytest suite repeated three times without contradictory outcomes;
- Python compilation of all new assets;
- deterministic cross-capability `v05_release_gate` PASS;
- GUM-5 Monte Carlo hard gates stable across multiple fixed seeds;
- randomized extreme-scale derivative verification;
- tiny-scale invalid covariance rejection in every new covariance-consuming path;
- structured/sparse path tested above dense-materialization scale;
- Gaussian joint-coverage empirical containment near its requested probability;
- time-series model requires explicit route and shows expected dependence effect;
- independent Sobol and correlated-Gaussian Shapley benchmark behavior;
- deterministic/input-sensitive reproducibility certificates;
- resource ceilings on potentially explosive dimensions/sample counts;
- unchanged v0.4.1 evidence inherited only when the exact parent artifacts remain untouched.

No independent reviewer is implied unless the host exposes a separate acceptance authority.
