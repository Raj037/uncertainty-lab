# v0.5 capability contract

## New capabilities

### Explicit joint distributions
`joint_distribution_engine.py` supports empirical/posterior joint samples, Gaussian/t copulas with explicit marginals, multivariate normal, and multivariate t. It does not infer a non-Gaussian joint distribution from covariance alone.

### Three independent derivative engines
`verification_engine.py` compares SymPy symbolic derivatives, `autodiff_engine.py` forward second-order AD, and an independent five-point AST finite-difference implementation. Comparison gates are dimensionless/scale-invariant.

### Structured covariance
`structured_covariance_engine.py` supports diagonal, diagonal-plus-low-rank, sparse factor, block diagonal, dense, and bounded arbitrary sparse COO. Large arbitrary sparse covariance is rejected unless PSD is proven by a supported factorized representation.

### Joint coverage
`coverage_regions.py` implements Gaussian Mahalanobis ellipsoids, empirical Mahalanobis regions, and empirical simultaneous boxes. Empirical regions are descriptive for the supplied sample; they are not automatically future-sample confidence guarantees.

### Explicit serial/repeatability models
`timeseries_engine.py` implements IID, stationary AR(1), Newey-West HAC, circular block bootstrap, and balanced random-intercept routes. The user/model must choose the route; diagnostics do not auto-select one.

### Attribution
`sensitivity_engine.py` uses Sobol first/total indices only for independent inputs and conditional-expectation Shapley effects for correlated multivariate-Gaussian inputs. Correlated non-Gaussian attribution is deliberately not claimed in v0.5.0.

## Retained v0.4.1 capabilities

v0.5.0 does not replace the qualified unit-aware, fit-provenance, latent-systematic, complex-linear, reference-engine, or dense published-covariance routes. They remain available through the unchanged v0.4.1 files inherited by the release.
