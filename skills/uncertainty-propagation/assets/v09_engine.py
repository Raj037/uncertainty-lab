#!/usr/bin/env python3
"""Unified v0.9 interface for Uncertainty Lab."""
from __future__ import annotations
import hashlib,json,platform,sys
from pathlib import Path
from typing import Any,Mapping
import numpy as np
import v08_engine as v08
import constrained_nuts_engine as cnuts
import pbox_engine as pbox
import time_reliability_engine as trel
import boolean_system_engine as bsys
import multifidelity_engine as mf
import global_design_engine as gdesign
import v09_release_gate as gate
ENGINE_VERSION='0.9.0'
EXPECTED_PARENT_HASHES={'v08_engine':'a3dc18037697ff2139c1f985c5e0efbc557f3fb67d5c45d57c839e1134dabc41','v08_release_gate':'d978e9850d569cc6f4f5ec09da743e41bb315e1cab7afac0d1a5994c2d865ce1','hmc_nuts_engine':'83cd4cdb41dbdd230fdd5b142163ac35241bef4ab77bd6322391ab5e6739a7bd','experimental_design_engine':'45814368b559461715a424e4877a94fba8eb7e0d0c89534d4606913868540a05','system_reliability_engine':'35851d1b8a8e937ac30f7d444c808ac66133a29ff8201c29542c486bd4f0f540','joint_distribution_engine':'6a0701edc88d3780232cd5f3207b95652ebba0ffc5c3899913b0750ab6d8bc2f'}
class InputError(ValueError):pass

def _sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def _sha_value(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def assert_parent_identity():
 mods={'v08_engine':v08,'v08_release_gate':__import__('v08_release_gate'),'hmc_nuts_engine':__import__('hmc_nuts_engine'),'experimental_design_engine':__import__('experimental_design_engine'),'system_reliability_engine':__import__('system_reliability_engine'),'joint_distribution_engine':__import__('joint_distribution_engine')};out={}
 for n,m in mods.items():
  a=_sha_file(Path(m.__file__));e=EXPECTED_PARENT_HASHES[n]
  if a!=e:raise InputError(f'v0.9 parent identity mismatch for {n}: {a}, expected {e}')
  out[n]=a
 return out
def make_certificate(payload,result):
 names=['v09_engine','constrained_nuts_engine','pbox_engine','time_reliability_engine','boolean_system_engine','multifidelity_engine','global_design_engine','v09_release_gate'];mods=[sys.modules[n] for n in names if n in sys.modules]
 src={m.__name__:_sha_file(m.__file__) for m in mods if getattr(m,'__file__',None)}
 c={'certificate_version':'7','engine':'uncertainty-lab-v09','engine_version':ENGINE_VERSION,'parent_identity':assert_parent_identity(),'payload_sha256':_sha_value(payload),'result_sha256':_sha_value(result),'source_sha256':src,'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':__import__('scipy').__version__,'sympy':__import__('sympy').__version__,'platform':platform.platform()}};c['certificate_sha256']=_sha_value(c);return c
def _certify(p,r):
 if isinstance(r,dict) and 'certificate' not in r:
  b=dict(r);r=dict(r);r['certificate']=make_certificate(p,b)
 return r
def run_request(req:Mapping[str,Any]):
 assert_parent_identity();op=str(req.get('operation','research_propagate'))
 if op=='constrained_nuts':out=cnuts.nuts(req)
 elif op=='pbox_cdf':out=pbox.cdf_bounds(req['pbox'],float(req['x']))
 elif op=='pbox_probability':out=pbox.probability_interval(req['pbox'],req.get('lower'),req.get('upper'))
 elif op=='pbox_propagate':out=pbox.propagate(req)
 elif op=='brownian_first_passage':out=trel.brownian(req)
 elif op=='ou_first_passage':out=trel.ou(req)
 elif op=='ar1_first_passage':out=trel.ar1(req)
 elif op=='boolean_system_reliability':out=bsys.evaluate(req)
 elif op=='multifidelity_control_variate':out=mf.control_variate(req)
 elif op=='multifidelity_gp':out=mf.autoregressive_gp(req)
 elif op=='experiment_design_global_linear':out=gdesign.exact_global_batch(req)
 elif op=='experiment_design_nonlinear_eig':out=gdesign.nonlinear_eig(req)
 elif op=='experiment_design_global_nonlinear_batch':out=gdesign.nonlinear_global_batch(req)
 elif op=='v09_release_gate':out=gate.run(int(req.get('seed',20261005)))
 else:return v08.run_request(req)
 return _certify(req,out)
def main():
 try:req=json.load(sys.stdin);print(json.dumps({'ok':True,'result':run_request(req)},indent=2,allow_nan=False));return 0
 except Exception as exc:print(json.dumps({'ok':False,'error':type(exc).__name__,'message':str(exc)},indent=2));return 2
if __name__=='__main__':raise SystemExit(main())
