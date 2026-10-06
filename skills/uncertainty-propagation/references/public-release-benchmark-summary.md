# Public-release benchmark summary — Uncertainty Lab v1.0.0

This benchmark was executed against the cleaned public-candidate source tree, not the private working directory.

**Result: 11/11 PASS.**

| Benchmark | Status |
|---|---:|
| `adequacy_nonrejection_not_proof` | **PASS** |
| `arviz_rank_rhat_ess` | **PASS** |
| `form_scale_invariance` | **PASS** |
| `jcgm_gum5_16_examples` | **PASS** |
| `missing_joint_law_fail_closed` | **PASS** |
| `nonlinear_complex_linear_consistency` | **PASS** |
| `openturns_cantilever_taylor` | **PASS** |
| `robust_decision_model_weight_uncertainty` | **PASS** |
| `structural_ensemble_total_variance_identity` | **PASS** |
| `tiny_indefinite_fail_closed` | **PASS** |
| `uqpy_form_linear_2d` | **PASS** |

## Interpretation

The benchmark demonstrates numerical agreement on selected published/reference cases plus fail-closed behavior on adversarial cases. It does **not** establish universal superiority over OpenTURNS, UQpy, PyMC, Stan, or other specialist software in performance, ecosystem maturity, solver depth, or every supported model class.

The strongest supported comparative claim is that Uncertainty Lab provides unusually broad integrated coverage with explicit fail-closed model/numerical checks in one self-contained workflow.

## Reproducibility

Exact machine-readable results are stored in `public-release-benchmark-results.json`. The public validation receipt records the source-tree identity, full test-suite result, multi-seed release-gate result, and static audit state.
