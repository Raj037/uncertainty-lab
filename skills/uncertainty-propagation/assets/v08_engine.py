#!/usr/bin/env python3
"""Unified v0.8 interface for Uncertainty Lab."""
from __future__ import annotations
import hashlib,json,platform,sys
from pathlib import Path
from typing import Any,Mapping
import numpy as np
import v07_engine as v07
import non_gaussian_reliability_engine as ngr
import hmc_nuts_engine as hmc
import discrepancy_inference_engine as discrepancy
import active_learning_reliability_engine as active
import multioutput_surrogate_engine as multiout
import system_reliability_engine as system
import experimental_design_engine as design
import v08_release_gate as gate
ENGINE_VERSION='0.8.0'
EXPECTED_PARENT_HASHES={'v07_engine':'e6030d2e9bd538323e4ec3e3422396f5221e32b49a8f6d11ff601a983b763962','v07_release_gate':'2ae7a63f1c5c52b1596fd54fdd1e66777993130f99342483c9e1db9ec2a458d8','reliability_engine':'e83013c10c1e7f1779cce6c11bb05f624e10e390db6f29405f0fdf692403e769','surrogate_engine':'6c11b93285f38342b34c4635e7f7582303720f3272fd4687ffceb5477e8b2b4e','calibration_engine':'7d7e15da1523205dd33ef2cf8f25b73f3e7bd162bd111fefe82d2f355b208068'}
class InputError(ValueError):pass
def _sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def _sha_value(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def assert_parent_identity():
 mods={'v07_engine':v07,'v07_release_gate':__import__('v07_release_gate'),'reliability_engine':__import__('reliability_engine'),'surrogate_engine':__import__('surrogate_engine'),'calibration_engine':__import__('calibration_engine')};out={}
 for n,m in mods.items():
  a=_sha_file(Path(m.__file__));e=EXPECTED_PARENT_HASHES[n]
  if a!=e:raise InputError(f'v0.8 parent identity mismatch for {n}: {a}, expected {e}')
  out[n]=a
 return out
def make_certificate(payload,result):
 names=['v08_engine','non_gaussian_reliability_engine','hmc_nuts_engine','discrepancy_inference_engine','active_learning_reliability_engine','multioutput_surrogate_engine','system_reliability_engine','experimental_design_engine','v08_release_gate'];mods=[sys.modules[n] for n in names if n in sys.modules]
 src={m.__name__:_sha_file(m.__file__) for m in mods if getattr(m,'__file__',None)}
 c={'certificate_version':'6','engine':'uncertainty-lab-v08','engine_version':ENGINE_VERSION,'parent_identity':assert_parent_identity(),'payload_sha256':_sha_value(payload),'result_sha256':_sha_value(result),'source_sha256':src,'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':__import__('scipy').__version__,'sympy':__import__('sympy').__version__,'platform':platform.platform()}};c['certificate_sha256']=_sha_value(c);return c
def _certify(p,r):
 if isinstance(r,dict) and 'certificate' not in r:
  b=dict(r);r=dict(r);r['certificate']=make_certificate(p,b)
 return r
def run_request(req:Mapping[str,Any]):
 assert_parent_identity();op=str(req.get('operation','research_propagate'))
 if op=='reliability_qmc_nongaussian':out=ngr.randomized_qmc(req)
 elif op=='reliability_directional':out=ngr.directional(req)
 elif op=='hmc_calibration':out=hmc.hmc(req)
 elif op=='nuts_calibration':out=hmc.nuts(req)
 elif op=='joint_discrepancy_inference':out=discrepancy.infer_linear_gaussian(req)
 elif op=='active_reliability_select':out=active.select_points(req)
 elif op=='active_reliability_expression':out=active.active_learn_expression(req)
 elif op=='multioutput_pce':out=multiout.pce(req)
 elif op=='multioutput_gp':out=multiout.gp_propagate(req)
 elif op=='system_reliability':out=system.randomized_qmc(req)
 elif op=='experiment_design_rank':out=design.score_candidates(req)
 elif op=='experiment_design_batch':out=design.greedy_batch(req)
 elif op=='v08_release_gate':out=gate.run(int(req.get('seed',20261005)))
 else:return v07.run_request(req)
 return _certify(req,out)
def main():
 try:req=json.load(sys.stdin);print(json.dumps({'ok':True,'result':run_request(req)},indent=2,allow_nan=False));return 0
 except Exception as exc:print(json.dumps({'ok':False,'error':type(exc).__name__,'message':str(exc)},indent=2));return 2
if __name__=='__main__':raise SystemExit(main())
