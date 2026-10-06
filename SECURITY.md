# Security and safety notes — Uncertainty Lab v1.0.0

Uncertainty Lab processes mathematical expressions through bounded/safe AST evaluators rather than unrestricted Python `eval`/`exec`. Numerical engines apply resource ceilings to high-cost sample, matrix, and expression paths where implemented.

The public candidate's static scan checks for:

- network-client imports and URLs in executable paths;
- subprocess/shell execution;
- embedded secrets or private keys;
- compiled/cache artifacts;
- unsafe expression execution patterns.

Scientific safety is handled separately from software security. The plugin deliberately fails closed on invalid covariance, undefined joint distributions, unresolved model assumptions, unsupported reliability transforms, unqualified surrogates, poor posterior diagnostics, or numerically unresolved calculations.

If a result affects safety-critical, regulated, clinical, financial, or contractual decisions, independent domain review and the applicable validation/accreditation process remain necessary.
