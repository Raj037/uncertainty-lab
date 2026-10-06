#!/usr/bin/env python3
"""Authoritative regression gates drawn from JCGM GUM-5:2026.

Only compact numerical data required for tests are stored. The benchmark explicitly does
not claim all 16 examples are implemented or that the software is formally certified.
"""
from __future__ import annotations
import math
from dataclasses import dataclass,asdict
from typing import Any
import numpy as np
import autodiff_engine as ad
import verification_engine as ver
import joint_distribution_engine as joint

SOURCE={"title":"JCGM GUM-5:2026 — Guide to the expression of uncertainty in measurement — Part 5: Examples",
        "doi":"10.59161/YNLY8209","official_publication":"https://www.bipm.org/en/committees/jc/jcgm/publications","edition":"First edition 2026"}
@dataclass(frozen=True)
class Check:
    name:str;observed:float;expected:float;atol:float;rtol:float=0.0
    def as_dict(self):
        e=abs(self.observed-self.expected);return {**asdict(self),"error":e,"passed":bool(e<=self.atol+self.rtol*abs(self.expected))}
def _out(cid,section,page,checks,notes=None):
    cs=[c.as_dict() for c in checks];return {"case_id":cid,"source":{**SOURCE,"section":section,"printed_page":page},"status":"PASS" if all(c['passed'] for c in cs) else "FAIL","checks":cs,"notes":list(notes or [])}
def _first_order(payload):
    r=ver.triple_verify(payload);C=np.asarray(r['symbolic_output_covariance']);
    # nominal/AD value is independent of propagation covariance
    vals=ad.differentiate(payload.get('expressions') or payload['expression'],payload['variables'],payload['values'])['values']
    return r,np.asarray(vals),np.sqrt(np.clip(np.diag(C),0,None))
def _second_order_sd(payload):
    ar=ad.differentiate(payload.get('expressions') or payload['expression'],payload['variables'],payload['values']);J=np.asarray(ar['jacobian']);Hs=np.asarray(ar['hessians']);C=np.asarray(payload['covariance']);F=J@C@J.T
    S=F.copy()
    for i,Hi in enumerate(Hs):
        for j,Hj in enumerate(Hs):S[i,j]+=0.5*np.trace(Hi@C@Hj@C)
    return np.sqrt(np.clip(np.diag(S),0,None))

def case_3_glucose():
    p={'variables':['H','C','O'],'values':[1.00798,12.0106,15.99940],'covariance':np.diag([.00008**2,.00058**2,.00021**2]).tolist(),'expression':'12*H+6*C+6*O'}
    v,y,u=_first_order(p);return _out('GUM5-3','3.3',21,[Check('three_engine_verification',1 if v['status']=='PASS' else 0,1,0),Check('relative_molecular_mass',float(y[0]),180.1557,1e-4),Check('standard_uncertainty',float(u[0]),.0038,5e-5)])
def case_6_2():
    p={'variables':['x1','x2','x3','x4'],'values':[0]*4,'covariance':np.eye(4).tolist(),'expression':'x1+x2+x3+x4'};v,y,u=_first_order(p)
    return _out('GUM5-6.2','6.2',42,[Check('three_engine_verification',1 if v['status']=='PASS' else 0,1,0),Check('estimate',float(y[0]),0,1e-12),Check('standard_uncertainty',float(u[0]),2,1e-12)])
def _uniform_case(scales,seed,n):
    marg=[{'kind':'uniform','lower':-math.sqrt(3)*s,'upper':math.sqrt(3)*s} for s in scales];d=len(scales)
    return joint.propagate({'variables':[f'x{i+1}' for i in range(d)],'joint_model':{'kind':'gaussian_copula','copula_correlation':np.eye(d).tolist(),'marginals':marg},'n':n,'seed':seed,'expression':'+'.join(f'x{i+1}' for i in range(d)),'coverage_probability':.95})
def case_6_3(seed=20261005,n=500000):
    r=_uniform_case([1,1,1,1],seed,n);ci=r['equal_tailed_coverage_interval'];q=2*math.sqrt(3)*(2-(3/5)**.25)
    return _out('GUM5-6.3','6.3',44,[Check('mc_mean',r['mean'][0],0,.015),Check('mc_standard_uncertainty',r['std'][0],2,.015),Check('coverage_lower',ci['lower'][0],-3.88,.04),Check('coverage_upper',ci['upper'][0],3.88,.04),Check('analytic_coverage_magnitude',q,3.88,.01)],[f'Deterministic PCG64 seed={seed}, N={n}.'])
def case_6_4(seed=20261005,n=500000):
    r=_uniform_case([1,1,1,10],seed,n);ci=r['equal_tailed_coverage_interval']
    return _out('GUM5-6.4','6.4',46,[Check('mc_mean',r['mean'][0],0,.05),Check('mc_standard_uncertainty',r['std'][0],10.1,.10),Check('coverage_lower',ci['lower'][0],-17,.15),Check('coverage_upper',ci['upper'][0],17,.15),Check('lpu_standard_uncertainty',math.sqrt(103),10.1,.05)])
def case_7():
    vals=[100000.,1.234,1.20,8000.,8000.];u=[.050,.020,.10/math.sqrt(3),1000/math.sqrt(3),.05/math.sqrt(3)]
    p={'variables':['mR','dmR','rhoa','rhoW','rhoR'],'values':vals,'covariance':np.diag(np.square(u)).tolist(),'expression':'(mR+dmR)*(1+(rhoa-1.2)*(1/rhoW-1/rhoR))-100000'}
    v,y,u1=_first_order(p);u2=_second_order_sd(p)
    return _out('GUM5-7','7.3',50,[Check('three_engine_verification',1 if v['status']=='PASS' else 0,1,0),Check('estimate',y[0],1.2340,5e-5),Check('first_order_standard_uncertainty',u1[0],.0539,5e-5),Check('second_order_standard_uncertainty',u2[0],.0750,1.5e-4)])
def case_14():
    p={'variables':['A0','T'],'values':[11.450,32.9],'covariance':[[.05**2,0],[0,1.4**2]],'expression':'A0*exp(-(2.73*log(2))/T)'};v,y,u=_first_order(p)
    return _out('GUM5-14','14.3',96,[Check('three_engine_verification',1 if v['status']=='PASS' else 0,1,0),Check('activity_kBq',y[0],10.810,.0006),Check('standard_uncertainty_kBq',u[0],.054,.0006)])

DOCUMENT_COVERAGE={
1:{'title':'Measurement of pH: linear interpolation','gate':'not_encoded'},2:{'title':'Determination of benzo[a]pyrene','gate':'not_encoded'},3:{'title':'Relative molecular mass of glucose','gate':'hard_gate'},4:{'title':'Gravimetric mixture preparation and calculation of composition','gate':'not_encoded'},5:{'title':'Greenhouse gas emission inventories','gate':'not_encoded'},6:{'title':'Simple linear measurement models','gate':'hard_gate','subcases':['6.2','6.3','6.4']},7:{'title':'Second-order effects in a nonlinear measurement model: calibration of weights','gate':'hard_gate'},8:{'title':'Gauge block calibration','gate':'not_encoded'},9:{'title':'GUM uncertainty evaluation for least-squares versus Bayesian inference — calibration of a torque measuring system','gate':'not_encoded'},10:{'title':'Conformity assessment of mass concentration of total suspended particulate matter in air','gate':'not_encoded'},11:{'title':'Effect of considering a 2D image as a set of pixels on a computed quantity','gate':'not_encoded'},12:{'title':'Between-bottle homogeneity of reference materials','gate':'not_encoded'},13:{'title':'Measurement of Celsius temperature using a resistance thermometer','gate':'not_encoded'},14:{'title':'Activity of a radioactive source corrected for decay','gate':'hard_gate'},15:{'title':'Breaking force of steel wire rope','gate':'not_encoded'},16:{'title':'Comparison loss in microwave power meter calibration','gate':'not_encoded'}}
def run_release_gate(seed=20261005,n=500000):
    cases=[case_3_glucose(),case_6_2(),case_6_3(seed,n),case_6_4(seed,n),case_7(),case_14()]
    return {'benchmark':'JCGM_GUM-5_2026_initial_gate','source':SOURCE,'status':'PASS' if all(x['status']=='PASS' for x in cases) else 'FAIL','hard_gate_count':len(cases),'cases':cases,'document_coverage':DOCUMENT_COVERAGE,'claim_boundary':'PASS qualifies only the six encoded hard gates; it is not a claim that all 16 GUM-5 examples are implemented or that formal GUM certification has been obtained.'}
