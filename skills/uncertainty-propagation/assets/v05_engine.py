#!/usr/bin/env python3
"""Unified v0.5 research interface for the new Uncertainty Lab capabilities."""
from __future__ import annotations
import json,math,sys,hashlib,platform
from typing import Any,Mapping
import numpy as np
import autodiff_engine as ad
import verification_engine as verify
import joint_distribution_engine as joint
import structured_covariance_engine as structured
import coverage_regions as coverage
import timeseries_engine as ts
import sensitivity_engine as sens
import gum5_benchmarks as gum5
import v05_release_gate as release_gate
ENGINE_VERSION='0.5.0'
class InputError(ValueError):pass


def _canonical_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
def _sha_value(value): return hashlib.sha256(_canonical_bytes(value)).hexdigest()
def _sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def make_certificate(payload,result):
    modules=[sys.modules[n] for n in ['v05_engine','autodiff_engine','verification_engine','joint_distribution_engine','structured_covariance_engine','coverage_regions','timeseries_engine','sensitivity_engine','gum5_benchmarks','v05_release_gate'] if n in sys.modules]
    src={m.__name__:_sha_file(m.__file__) for m in modules if getattr(m,'__file__',None)}
    cert={'certificate_version':'3','engine':'uncertainty-lab-v05','engine_version':ENGINE_VERSION,'payload_sha256':_sha_value(payload),'result_sha256':_sha_value(result),'source_sha256':src,
          'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':__import__('scipy').__version__,'sympy':__import__('sympy').__version__,'platform':platform.platform()}}
    cert['certificate_sha256']=_sha_value(cert);return cert
def _certify(payload,result):
    if isinstance(result,dict) and 'certificate' not in result:
        base=dict(result);result=dict(result);result['certificate']=make_certificate(payload,base)
    return result

def _exprs(p):
    x=list(p.get('expressions') or ([p['expression']] if 'expression' in p else []))
    if not x:raise InputError('expression or expressions required')
    return x

def research_propagate(payload:Mapping[str,Any])->dict[str,Any]:
    """Plain-real Gaussian covariance workflow with three-engine verification.

    Unit-aware, fit-provenance and complex-linear v0.4.1 workflows remain available in
    their established engines; this route is the v0.5 three-engine numerical core.
    """
    p=dict(payload);names=list(p.get('variables') or []);mu=np.asarray(p.get('values'),dtype=float);C=np.asarray(p.get('covariance'),dtype=float);expr=_exprs(p)
    try:v=verify.triple_verify(p)
    except Exception as exc:return {'engine_version':ENGINE_VERSION,'status':'BLOCKED','issue':'MODEL_OR_VERIFICATION_ERROR','message':str(exc),'verification':None}
    if v['status']!='PASS':return {'engine_version':ENGINE_VERSION,'status':'BLOCKED','issue':'THREE_ENGINE_DISAGREEMENT','verification':v,'analytic':None}
    d=ad.differentiate(expr,names,mu);J=np.asarray(d['jacobian']);Hs=np.asarray(d['hessians']);y=np.asarray(d['values']);first=J@C@J.T;first=(first+first.T)/2
    first_sd=np.sqrt(np.clip(np.diag(first),0,None));shifts=np.array([.5*np.trace(H@C) for H in Hs]);second=first.copy()
    for i,Hi in enumerate(Hs):
        for j,Hj in enumerate(Hs):second[i,j]+=.5*np.trace(Hi@C@Hj@C)
    second=(second+second.T)/2;second_sd=np.sqrt(np.clip(np.diag(second),0,None));shift_sigma=np.divide(shifts,first_sd,out=np.full_like(shifts,np.nan),where=first_sd>0);ratio=np.divide(second_sd,first_sd,out=np.full_like(second_sd,np.nan),where=first_sd>0)
    triggers=[]
    for i in range(len(expr)):
        if np.isfinite(shift_sigma[i]) and abs(shift_sigma[i])>=.1:triggers.append({'output':i,'metric':'mean_shift_sigma','value':float(shift_sigma[i])})
        if np.isfinite(ratio[i]) and abs(ratio[i]-1)>=.05:triggers.append({'output':i,'metric':'second_to_first_std_ratio','value':float(ratio[i])})
    status='PROVISIONAL' if triggers else 'MODEL_READY'
    out={'engine_version':ENGINE_VERSION,'status':status,'method':'three_engine_Gaussian_first_second_order','verification':v,
         'analytic':{'nominal':y.tolist(),'jacobian':J.tolist(),'hessians':Hs.tolist(),'first_order_output_covariance':first.tolist(),'first_order_std_uncertainty':first_sd.tolist(),'second_order_mean_shift':shifts.tolist(),'second_order_output_covariance':second.tolist(),'second_order_std_uncertainty':second_sd.tolist(),'nonlinearity_triggers':triggers},
         'claim_boundary':'Three-engine agreement verifies numerical derivatives/propagation, not the scientific validity of the supplied measurement model.'}
    if p.get('coverage_probability') is not None:
        out['gaussian_coverage_region']=coverage.gaussian_ellipsoid(y,first,float(p['coverage_probability']))
    return out

def run_request(req:Mapping[str,Any])->dict[str,Any]:
    op=str(req.get('operation','research_propagate'))
    if op=='research_propagate':out=research_propagate(req)
    elif op=='triple_verify':out=verify.triple_verify(req)
    elif op=='joint_propagate':out=joint.propagate(req)
    elif op=='structured_covariance':out=structured.linear_propagate(req['covariance_model'],req['linear_map'])
    elif op=='gaussian_coverage_region':out=coverage.gaussian_ellipsoid(req['mean'],req['covariance'],float(req.get('probability',.95)))
    elif op=='empirical_coverage_region':out=coverage.empirical_ellipsoid(req['samples'],float(req.get('probability',.95)))
    elif op=='empirical_joint_box':out=coverage.empirical_maxnorm_box(req['samples'],float(req.get('probability',.95)))
    elif op=='timeseries':out=ts.analyze(req)
    elif op=='sobol_independent':out=sens.sobol_independent(req)
    elif op=='shapley_correlated_gaussian':out=sens.shapley_correlated_gaussian(req)
    elif op=='gum5_gate':out=gum5.run_release_gate(int(req.get('seed',20261005)),int(req.get('n',500000)))
    elif op=='v05_release_gate':out=release_gate.run(int(req.get('seed',20261005)))
    else:raise InputError(f'unknown v0.5 operation {op!r}')
    return _certify(req,out)

def main()->int:
    try:
        req=json.load(sys.stdin);print(json.dumps({'ok':True,'result':run_request(req)},indent=2,allow_nan=False));return 0
    except Exception as e:
        print(json.dumps({'ok':False,'error':type(e).__name__,'message':str(e)},indent=2));return 2
if __name__=='__main__':raise SystemExit(main())
