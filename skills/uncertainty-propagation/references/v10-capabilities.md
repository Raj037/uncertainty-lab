# v1.0 capability contract

## Structural model uncertainty
`model_ensemble_engine.py` combines an explicit candidate model set using supplied weights or prior-adjusted log evidences. It reports mixture mean, within-model covariance, between-model covariance, and total covariance separately. Optional model-weight intervals provide robust scalar bounds through linear programming. The candidate model set is never asserted complete merely because the mixture is numerically well-defined.

## Robust decisions
`robust_decision_engine.py` evaluates a finite action/scenario loss matrix over an explicit feasible model-weight polytope. It distinguishes nominal Bayes action, minimax expected-loss action, minimax-regret action, and maximin experiment utility. Guarantees do not extend to omitted scenarios or misspecified losses/utilities.

## Nonlinear complex propagation
`nonlinear_complex_engine.py` supports safe nonlinear complex expressions with full real/imag multivariate-normal input covariance. It returns Monte Carlo real/imag covariance, Hermitian covariance, pseudo-covariance, magnitude summaries, circular phase summaries, invalid-domain fraction, and a local real-coordinate five-point linearization for comparison. Phase is treated as circular and may be uninformative when resultant length is small.
