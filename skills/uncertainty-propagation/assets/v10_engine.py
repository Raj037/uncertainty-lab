#!/usr/bin/env python3
"""Unified v1.0 interface for Uncertainty Lab."""
from __future__ import annotations
import hashlib,json,platform,sys
from pathlib import Path
from typing import Any,Mapping
import numpy as np
import v09_engine as v09
import model_ensemble_engine as ensemble
import robust_decision_engine as robust
import nonlinear_complex_engine as cplx
import v10_release_gate as gate
ENGINE_VERSION='1.0.0'
EXPECTED_PARENT_HASHES={'v09_engine':'211d62825ff02d18d02135729833b90c8046f717f74c68967be795d52ce699a2','v09_release_gate':'6bc3af02085b32e1665239e3a3ff1b1067b91f261b143686dc8feaef308ab389','constrained_nuts_engine':'3e1315977035ee9f4888223f388b053df0998ffef4e713d58d739f887a332985','pbox_engine':'50eaaffb182c1a15d9e7d29f6bb3d3bddf9756b071fc727f327c757d260ffeda','time_reliability_engine':'797c86f6860c8e0c9ee2866109de40a03debcbcce394eb1e9d75301d32daf621','boolean_system_engine':'d43f7e3cfd0dc34ab34ed32e7f3e8c1ccde0b37c4a65848c3210992944507f7f','multifidelity_engine':'55b58ff6fc0db08474c18e4d1a9aa40ea97041494c08f3197af19365811a978d','global_design_engine':'9b07822585156e1cba1de6902a8e4e0f4fd417b82ffc549e9919ef51e4e927c3'}
class InputError(ValueError):pass
def _sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def _sha_value(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def assert_parent_identity():
 mods={'v09_engine':v09,'v09_release_gate':__import__('v09_release_gate'),'constrained_nuts_engine':__import__('constrained_nuts_engine'),'pbox_engine':__import__('pbox_engine'),'time_reliability_engine':__import__('time_reliability_engine'),'boolean_system_engine':__import__('boolean_system_engine'),'multifidelity_engine':__import__('multifidelity_engine'),'global_design_engine':__import__('global_design_engine')};out={}
 for n,m in mods.items():
  a=_sha_file(Path(m.__file__));e=EXPECTED_PARENT_HASHES[n]
  if a!=e:raise InputError(f'v1.0 parent identity mismatch for {n}: {a}, expected {e}')
  out[n]=a
 return out
def make_certificate(payload,result):
 names=['v10_engine','model_ensemble_engine','robust_decision_engine','nonlinear_complex_engine','v10_release_gate'];mods=[sys.modules[n] for n in names if n in sys.modules]
 src={m.__name__:_sha_file(m.__file__) for m in mods if getattr(m,'__file__',None)}
 c={'certificate_version':'8','engine':'uncertainty-lab-v10','engine_version':ENGINE_VERSION,'parent_identity':assert_parent_identity(),'payload_sha256':_sha_value(payload),'result_sha256':_sha_value(result),'source_sha256':src,'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':__import__('scipy').__version__,'sympy':__import__('sympy').__version__,'platform':platform.platform()}};c['certificate_sha256']=_sha_value(c);return c
def _certify(p,r):
 if isinstance(r,dict) and 'certificate' not in r:
  b=dict(r);r=dict(r);r['certificate']=make_certificate(p,b)
 return r
def run_request(req:Mapping[str,Any]):
 assert_parent_identity();op=str(req.get('operation','research_propagate'))
 if op=='model_ensemble':out=ensemble.combine(req)
 elif op=='model_weight_robust_bounds':out=ensemble.robust_scalar_bounds(req)
 elif op=='robust_decision':out=robust.analyze(req)
 elif op=='robust_experiment_design':out=robust.robust_utility_design(req)
 elif op=='nonlinear_complex_propagate':out=cplx.propagate(req)
 elif op=='v10_release_gate':out=gate.run(int(req.get('seed',20261005)))
 else:return v09.run_request(req)
 return _certify(req,out)
def main():
 try:req=json.load(sys.stdin);print(json.dumps({'ok':True,'result':run_request(req)},indent=2,allow_nan=False));return 0
 except Exception as exc:print(json.dumps({'ok':False,'error':type(exc).__name__,'message':str(exc)},indent=2));return 2
if __name__=='__main__':raise SystemExit(main())
