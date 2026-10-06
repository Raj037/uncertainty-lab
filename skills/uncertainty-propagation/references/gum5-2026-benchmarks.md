# JCGM GUM-5:2026 regression gates

Primary source: JCGM GUM-5:2026, *Guide to the expression of uncertainty in measurement — Part 5: Examples*, first edition 2026, DOI `10.59161/YNLY8209`, BIPM/JCGM.

The v0.5.0 release gate encodes six numerical hard gates:

1. relative molecular mass of glucose — linear LPU / Type-B-style input uncertainties;
2. simple linear model with four normal inputs;
3. simple linear model with four equal rectangular inputs — Monte Carlo and analytic coverage;
4. simple linear model with one dominant rectangular input — demonstration that a Gaussian/CLT coverage shortcut can be misleading;
5. calibration of weights — first- versus second-order propagation;
6. radioactive-source activity corrected for decay — nonlinear exponential propagation.

The benchmark module also contains a 16-example topic coverage matrix. Ten examples are explicitly `not_encoded`; they must not be counted as PASS.

Release qualification language must therefore be: **“six encoded GUM-5 hard gates pass”**, not “all GUM-5 examples pass” and not “formally GUM-certified.”
