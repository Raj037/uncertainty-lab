#!/usr/bin/env python3
"""Deterministic cross-capability release gate for Uncertainty Lab v0.5.0."""
from __future__ import annotations
import json, math
from typing import Any
import numpy as np

import verification_engine as verify
import joint_distribution_engine as joint
import structured_covariance_engine as structured
import coverage_regions as coverage
import timeseries_engine as ts
import sensitivity_engine as sens
import gum5_benchmarks as gum5

ENGINE_VERSION="0.5.0"


def _row(name:str, passed:bool, evidence:Any=None)->dict[str,Any]:
    return {"name":name,"status":"PASS" if passed else "FAIL","evidence":evidence}


def run(seed:int=20261005)->dict[str,Any]:
    rows=[]
    # 1. Official GUM-5 gates, repeated across independent deterministic streams.
    gum=[]
    for s in (1,1729,seed):
        r=gum5.run_release_gate(seed=s,n=500000)
        gum.append({"seed":s,"status":r["status"]})
    rows.append(_row("gum5_multiseed",all(x["status"]=="PASS" for x in gum),gum))

    # 2. Three-engine verification under extreme positive input rescaling.
    rng=np.random.default_rng(seed)
    vfails=0
    for _ in range(150):
        n=int(rng.integers(1,6)); scales=10.0**rng.uniform(-20,20,size=n)
        A=rng.normal(size=(n,n)); base=A@A.T+np.eye(n)*.2;D=np.diag(scales);C=D@base@D
        mu=rng.normal(size=n)*scales; a=rng.normal(size=n)
        expr="+".join(f"({a[i]/scales[i]:.17g})*x{i}" for i in range(n))
        p={"variables":[f"x{i}" for i in range(n)],"values":mu.tolist(),"covariance":C.tolist(),"expression":expr}
        if verify.triple_verify(p)["status"]!="PASS":vfails+=1
    rows.append(_row("three_engine_extreme_rescaling",vfails==0,{"cases":150,"failures":vfails,"scale_per_axis":"10^-20..10^20"}))

    # 3. Explicit correlated non-Gaussian joint propagation.
    jp={"variables":["x","y"],"joint_model":{"kind":"t_copula","df":4,"copula_correlation":[[1,.65],[.65,1]],
        "marginals":[{"kind":"uniform","lower":-2,"upper":2},{"kind":"lognormal","log_mean":0,"log_std":.45}]},
        "n":120000,"seed":seed,"expressions":["x+y","x*y"],"coverage_probability":.95}
    jr=joint.propagate(jp); jc=np.asarray(jr["output_covariance"])
    rows.append(_row("correlated_nongaussian_joint",jr["status"]=="ok" and jc.shape==(2,2) and np.linalg.eigvalsh(jc).min()>-1e-10,
                     {"n":jr["n"],"invalid_fraction":jr["invalid_fraction"],"joint_kind":jr["joint_model"]["kind"]}))

    # 4. Large structured factor path without dense materialization.
    n=12000;q=4
    rows_i=np.arange(n);cols_i=rows_i%q;data=np.linspace(.001,.004,n)
    spec={"kind":"sparse_factor","diagonal":np.full(n,1e-4).tolist(),"factor":{"shape":[n,q],"rows":rows_i.tolist(),"cols":cols_i.tolist(),"data":data.tolist()}}
    J=np.zeros((2,n));J[0,:]=1/n;J[1,::2]=2/n
    sr=structured.linear_propagate(spec,J); sc=np.asarray(sr["output_covariance"])
    rows.append(_row("structured_covariance_12k",sc.shape==(2,2) and np.linalg.eigvalsh(sc).min()>-1e-15,
                     {"input_dimension":sr["input_dimension"],"representation":sr["representation"]}))

    # 5. Multivariate Gaussian coverage region empirical check.
    C=np.array([[1,.35,-.1],[.35,2,.25],[-.1,.25,.7]],float);mu=np.array([2.,-1.,.5])
    reg=coverage.gaussian_ellipsoid(mu,C,.95);X=rng.multivariate_normal(mu,C,size=100000)
    frac=float(np.mean(coverage.contains(reg,X)))
    rows.append(_row("gaussian_joint_coverage",abs(frac-.95)<.006,{"empirical_containment":frac,"target":.95,"n":100000}))

    # 6. Explicit time-series model; dependence must inflate mean uncertainty.
    phi=.72;eps=rng.normal(size=8000);x=np.empty(8000);x[0]=eps[0]
    for i in range(1,len(x)):x[i]=phi*x[i-1]+eps[i]
    iid=ts.iid_mean(x);ar=ts.ar1_mean(x,phi=phi)
    rows.append(_row("explicit_timeseries_dependence",ar["standard_uncertainty"]>iid["standard_uncertainty"]*1.8,
                     {"iid_u":iid["standard_uncertainty"],"ar1_u":ar["standard_uncertainty"],"phi":phi}))

    # 7. Nonlinear sensitivity: independent Sobol and correlated-Gaussian Shapley.
    sob=sens.sobol_independent({"variables":["a","b"],"marginals":[{"kind":"uniform","lower":-1,"upper":1},{"kind":"uniform","lower":-1,"upper":1}],
        "expression":"a+2*b","n":40000,"replicates":3,"seed":seed})
    f=np.asarray(sob["first_order_mean"]);sob_ok=np.allclose(f,[.2,.8],atol=.035)
    sha=sens.shapley_correlated_gaussian({"variables":["x","y"],"mean":[0,0],"covariance":[[1,.6],[.6,1]],"expression":"x+y",
        "outer_samples":700,"inner_samples":120,"replicates":2,"seed":seed})
    sh=np.asarray(sha["effects_mean"]);sh_ok=abs(sh.sum()-1)<.08 and np.all(sh>0)
    rows.append(_row("nonlinear_sensitivity_attribution",sob_ok and sh_ok,{"sobol_first":f.tolist(),"shapley":sh.tolist(),"shapley_sum":float(sh.sum())}))

    # Cross-cutting adversarial regression: tiny indefinite covariance is rejected by joint MVN.
    blocked=False
    try:
        joint.propagate({"variables":["x","y"],"joint_model":{"kind":"multivariate_normal","mean":[0,0],"covariance":[[1e-40,2e-40],[2e-40,1e-40]]},"n":100,"expression":"x+y"})
    except joint.InputError:
        blocked=True
    rows.append(_row("tiny_indefinite_fail_closed",blocked))

    status="PASS" if all(r["status"]=="PASS" for r in rows) else "FAIL"
    return {"engine_version":ENGINE_VERSION,"gate":"v0.5_cross_capability_release_gate","status":status,"seed":seed,"rows":rows,
            "claim_boundary":"This gate verifies the encoded software capabilities and six official GUM-5 hard gates; it is not formal metrology accreditation or proof of a user-supplied scientific model."}

if __name__=="__main__":
    r=run();print(json.dumps(r,indent=2,allow_nan=False));raise SystemExit(0 if r["status"]=="PASS" else 2)
