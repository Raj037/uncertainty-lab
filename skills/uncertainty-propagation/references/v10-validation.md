# v1.0 validation notes

- Model-ensemble benchmark with two equally weighted models reproduces mixture mean 1, within-model variance 1, between-model variance 1, total variance 2.
- Robust-decision benchmark intentionally separates the nominal Bayes action from the minimax expected-loss action under model-weight uncertainty.
- Nonlinear complex linear-map control reproduces the exact real-coordinate covariance in local linearization and Monte Carlo covariance within the preregistered stochastic tolerance.
- v1.0-specific suite: 18/18 PASS.
- Integrated v1.0 release gate passed for seeds 1, 1729, 20261005, 8675309, and 2147483647; every run nests the complete v0.9 release gate.

Claim boundary: this is plugin-only software verification, not proof that a candidate model set is complete, not a universal robust-decision guarantee, and not formal metrology/reliability accreditation.
