#!/usr/bin/env python3
"""Integrated deterministic release gate for Uncertainty Lab v0.7.0."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

import v06_release_gate as v06_gate
import model_adequacy_engine as adequacy
import reliability_engine as reliability
import surrogate_engine as surrogate
import calibration_engine as calibration
import conformity_engine as conformity

ENGINE_VERSION = "0.7.0"


def _row(name: str, passed: bool, evidence: Any = None) -> dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "evidence": evidence}


def run(seed: int = 20261005) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    # 1. Entire v0.6 integrated release gate remains green.
    parent=v06_gate.run(seed)
    rows.append(_row("parent_v06_release_gate", parent["status"]=="PASS", {"status":parent["status"],"rows":parent["rows"]}))

    # 2. Model adequacy: good residuals are only non-rejected; strong bias is flagged;
    #    an unresolved material effect prevents MODEL_READY even with tiny residuals.
    good=adequacy.assess({"observed":[.05,-.04,.02,.01,-.03,.02],"predicted":[0]*6,"standard_uncertainties":[.2]*6,
                          "effects":[{"name":"temperature","status":"included","material":True}],"effect_register_complete":True,"effect_register_justification":"measurement procedure and influence-quantity review completed"})
    bad=adequacy.assess({"observed":[2.0]*8,"predicted":[0]*8,"standard_uncertainties":[.2]*8,"effects":[]})
    unresolved=adequacy.assess({"observed":[.01,-.01,.0,.01],"predicted":[0]*4,"standard_uncertainties":[.1]*4,
                                "effects":[{"name":"humidity","status":"unresolved","material":True}]})
    ok=good["adequacy_conclusion"]=="ADEQUACY_NOT_REJECTED_NOT_PROVEN" and bad["adequacy_conclusion"]=="MODEL_INADEQUACY_EVIDENCE" and unresolved["status"]=="PROVISIONAL"
    rows.append(_row("model_adequacy_fail_closed",ok,{"good":good["adequacy_conclusion"],"bad_p":bad["residual_diagnostics"]["chi_square_pvalue"],"unresolved":unresolved["adequacy_conclusion"]}))

    # 3. Reliability: FORM exact for a linear normal limit state; importance sampling
    #    resolves a 4-sigma tail; subset simulation independently tracks a 3.5-sigma tail.
    base={"variables":["x"],"mean":[0.0],"covariance":[[1.0]],"limit_state":"4-x"}
    f=reliability.form(base); s=reliability.sorm(base); imp=reliability.importance_sampling({**base,"n":60000,"seed":seed})
    sub=reliability.subset_simulation({"variables":["x"],"mean":[0.0],"covariance":[[1.0]],"limit_state":"3.5-x","n":2500,"p0":.1,"replicates":3,"seed":seed})
    exact4=float(stats.norm.cdf(-4));exact35=float(stats.norm.cdf(-3.5));sub_ratio=sub["failure_probability_mean"]/exact35
    rel_ok=abs(f["failure_probability_FORM"]-exact4)<1e-10 and s["failure_probability_SORM_Breitung"]==f["failure_probability_FORM"] and abs(imp["failure_probability"]-exact4)<5*imp["standard_error"] and .4<sub_ratio<2.5
    rows.append(_row("rare_event_reliability",rel_ok,{"FORM":f["failure_probability_FORM"],"exact4":exact4,"IS":imp["failure_probability"],"IS_se":imp["standard_error"],"subset_ratio_to_exact":sub_ratio}))

    # 4. Surrogates: an exact quadratic PCE qualifies with analytic moments; GP
    #    qualifies on held-out sine data. Validation is the acceptance authority.
    rng=np.random.default_rng(seed);X=rng.uniform(-1,1,size=(150,2));y=1+2*X[:,0]+3*X[:,1]**2
    pce=surrogate.polynomial_chaos({"X":X.tolist(),"y":y.tolist(),"marginals":[{"kind":"uniform","lower":-1,"upper":1}]*2,"total_degree":2,
                                    "validation_fraction":.25,"validation_seed":seed,"qualification":{"max_nrmse":1e-9,"min_r2":.99999999}})
    gx=np.linspace(-3,3,80)[:,None];gy=np.sin(gx[:,0])
    gp=surrogate.gaussian_process({"X":gx.tolist(),"y":gy.tolist(),"validation_fraction":.2,"validation_seed":seed,"noise_std":1e-6,
                                   "qualification":{"max_nrmse":.03,"min_r2":.99}})
    sur_ok=pce["status"]=="SURROGATE_QUALIFIED" and abs(pce["distribution_mean_from_coefficients"]-2)<1e-8 and abs(pce["distribution_variance_from_coefficients"]-(4/3+.8))<1e-8 and gp["status"]=="SURROGATE_QUALIFIED"
    rows.append(_row("validated_surrogates",sur_ok,{"PCE_rmse":pce["validation_metrics"]["rmse"],"PCE_var":pce["distribution_variance_from_coefficients"],"GP_r2":gp["validation_metrics"]["r2"]}))

    # 5. Calibration: exact linear-Gaussian posterior and multi-chain MCMC must agree;
    #    adding explicit discrepancy covariance must widen the posterior.
    yy=[1.0,1.2,.8,1.1,.9,1.05,.95,1.1];A=np.ones((len(yy),1))
    exact=calibration.linear_gaussian({"observed":yy,"design_matrix":A.tolist(),"prior_mean":[0.0],"prior_covariance":[[1.0]],"observation_sd":.5})
    disc=calibration.linear_gaussian({"observed":yy,"design_matrix":A.tolist(),"prior_mean":[0.0],"prior_covariance":[[1.0]],"observation_sd":.5,"discrepancy_covariance":(.3**2*np.eye(len(yy))).tolist()})
    mcmc=calibration.metropolis({"parameter_names":["theta"],"priors":{"theta":{"kind":"normal","mean":0,"std":1}},"observed":yy,"observation_sd":.5,
                                 "model_expression":"theta","chains":4,"warmup":900,"draws":1700,"proposal_sd":[.18],"seed":seed,"min_bulk_ess":250,"min_tail_ess":250})
    cal_ok=mcmc["status"]=="POSTERIOR_QUALIFIED" and abs(mcmc["posterior_mean"][0]-exact["posterior_mean"][0])<.05 and abs(mcmc["posterior_sd"][0]-exact["posterior_sd"][0])<.035 and disc["posterior_sd"][0]>exact["posterior_sd"][0]
    rows.append(_row("bayesian_calibration_with_discrepancy",cal_ok,{"exact_mean":exact["posterior_mean"][0],"mcmc_mean":mcmc["posterior_mean"][0],"exact_sd":exact["posterior_sd"][0],"mcmc_sd":mcmc["posterior_sd"][0],"discrepancy_sd":disc["posterior_sd"][0],"diagnostics":mcmc["diagnostics"]["status"]}))

    # 6. Conformity: one-sided normal guard band must equal the exact normal quantile;
    #    item-specific risk and global population risk remain distinct outputs.
    guard=conformity.design_guard_band({"tolerance_upper":10,"standard_uncertainty":.2,"max_specific_consumer_risk":.025})
    expected_w=float(stats.norm.ppf(.975)*.2)
    item=conformity.decide({"tolerance_upper":10,"acceptance_upper":10-expected_w,"measured_value":9.5,"measurement_distribution":{"kind":"normal","mean":9.5,"std":.2}})
    glob=conformity.global_risk({"tolerance_lower":-1,"tolerance_upper":1,"acceptance_lower":-.8,"acceptance_upper":.8,"n":60000,"seed":seed,
                                 "true_distribution":{"kind":"normal","mean":0,"std":1},"measurement_error_distribution":{"kind":"normal","mean":0,"std":.2}})
    conf_ok=abs(guard["guard_band_width"]-expected_w)<1e-12 and abs(guard["consumer_risk_at_active_acceptance_boundary"]-float(stats.norm.cdf(-stats.norm.ppf(.975))))<1e-12 and item["decision"]=="ACCEPT" and 0<glob["global_consumer_risk_joint_probability_false_accept"]<.1
    rows.append(_row("conformity_and_decision_risk",conf_ok,{"guard_width":guard["guard_band_width"],"boundary_risk":guard["consumer_risk_at_active_acceptance_boundary"],"item_consumer_risk":item["specific_consumer_risk_if_accepted"],"global_consumer_risk":glob["global_consumer_risk_joint_probability_false_accept"]}))

    status="PASS" if all(r["status"]=="PASS" for r in rows) else "FAIL"
    return {"engine_version":ENGINE_VERSION,"gate":"v0.7_integrated_release_gate","status":status,"seed":int(seed),"rows":rows,
            "claim_boundary":"The gate validates encoded numerical capabilities and preserves the complete v0.6 parent gate. It is not formal metrology accreditation, reliability certification, surrogate-model universal validity, or authorization of a user-specific conformity decision."}

if __name__=="__main__":
    import json
    z=run();print(json.dumps(z,indent=2,allow_nan=False));raise SystemExit(0 if z["status"]=="PASS" else 2)
