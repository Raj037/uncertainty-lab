#!/usr/bin/env python3
"""Unified v0.7 interface for Uncertainty Lab."""
from __future__ import annotations

import hashlib,json,platform,sys
from pathlib import Path
from typing import Any,Mapping
import numpy as np

import v06_engine as v06
import model_adequacy_engine as adequacy
import reliability_engine as reliability
import surrogate_engine as surrogate
import calibration_engine as calibration
import conformity_engine as conformity
import v07_release_gate as release_gate

ENGINE_VERSION="0.7.0"
EXPECTED_PARENT_HASHES={
    "v06_engine":"dc4bd8408579bfdbaf76d879d69fcc1d32035c06fc14a141aaa684c6ddb6d99b",
    "v06_release_gate":"302e671bd8ab8fd43fe00d0d4c65d3904ccf043218550108cdb6267b3f00c9f5",
    "joint_distribution_engine":"6a0701edc88d3780232cd5f3207b95652ebba0ffc5c3899913b0750ab6d8bc2f",
    "autodiff_engine":"b24750dd5d3bda86efd15d160ba51b52c060972c0878bce3fda992cb058b3606",
    "posterior_diagnostics":"6006b8c2d143963c69150b9be2592fb268f099d7b1b7abce282e7f49b0a8fa37",
}
class InputError(ValueError):pass

def _sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def _sha_value(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def assert_parent_identity():
    mods={"v06_engine":v06,"v06_release_gate":__import__('v06_release_gate'),"joint_distribution_engine":__import__('joint_distribution_engine'),"autodiff_engine":__import__('autodiff_engine'),"posterior_diagnostics":__import__('posterior_diagnostics')};got={}
    for name,m in mods.items():
        actual=_sha_file(Path(m.__file__));expected=EXPECTED_PARENT_HASHES[name]
        if actual!=expected:raise InputError(f'v0.7 parent identity mismatch for {name}: {actual}, expected {expected}')
        got[name]=actual
    return got

def make_certificate(payload,result):
    names=['v07_engine','model_adequacy_engine','reliability_engine','surrogate_engine','calibration_engine','conformity_engine','v07_release_gate'];mods=[sys.modules[n] for n in names if n in sys.modules]
    src={m.__name__:_sha_file(m.__file__) for m in mods if getattr(m,'__file__',None)}
    cert={'certificate_version':'5','engine':'uncertainty-lab-v07','engine_version':ENGINE_VERSION,'parent_identity':assert_parent_identity(),
          'payload_sha256':_sha_value(payload),'result_sha256':_sha_value(result),'source_sha256':src,
          'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':__import__('scipy').__version__,'sympy':__import__('sympy').__version__,'platform':platform.platform()}}
    cert['certificate_sha256']=_sha_value(cert);return cert
def _certify(p,r):
    if isinstance(r,dict) and 'certificate' not in r:
        b=dict(r);r=dict(r);r['certificate']=make_certificate(p,b)
    return r

def run_request(req:Mapping[str,Any])->dict[str,Any]:
    assert_parent_identity();op=str(req.get('operation','research_propagate'))
    if op=='model_adequacy':out=adequacy.assess(req)
    elif op=='residual_diagnostics':out=adequacy.residual_diagnostics(req)
    elif op=='model_discrepancy_diagnostic':out=adequacy.estimate_additive_discrepancy(req)
    elif op=='compare_models':out=adequacy.compare_models(req)
    elif op=='FORM':out=reliability.form(req)
    elif op=='SORM':out=reliability.sorm(req)
    elif op=='reliability_importance_sampling':out=reliability.importance_sampling(req)
    elif op=='subset_simulation':out=reliability.subset_simulation(req)
    elif op=='reliability_all':out=reliability.run_all(req)
    elif op=='gaussian_process_surrogate':out=surrogate.gaussian_process(req)
    elif op=='pce_surrogate':out=surrogate.polynomial_chaos(req)
    elif op=='propagate_gp_surrogate':out=surrogate.propagate_gp(req)
    elif op=='propagate_pce_surrogate':out=surrogate.propagate_pce(req)
    elif op=='linear_gaussian_calibration':out=calibration.linear_gaussian(req)
    elif op=='mcmc_calibration':out=calibration.metropolis(req)
    elif op=='conformance_probability':out=conformity.conformance_probability(req)
    elif op=='conformity_decision':out=conformity.decide(req)
    elif op=='guard_band_design':out=conformity.design_guard_band(req)
    elif op=='global_decision_risk':out=conformity.global_risk(req)
    elif op=='v07_release_gate':out=release_gate.run(int(req.get('seed',20261005)))
    else:return v06.run_request(req)
    return _certify(req,out)

def main():
    try:req=json.load(sys.stdin);print(json.dumps({'ok':True,'result':run_request(req)},indent=2,allow_nan=False));return 0
    except Exception as exc:print(json.dumps({'ok':False,'error':type(exc).__name__,'message':str(exc)},indent=2));return 2
if __name__=='__main__':raise SystemExit(main())
