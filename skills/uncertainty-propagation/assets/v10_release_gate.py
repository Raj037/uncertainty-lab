#!/usr/bin/env python3
"""Integrated final broad-scope release gate for Uncertainty Lab v1.0.0."""
from __future__ import annotations
import numpy as np
import v09_release_gate as parent
import model_ensemble_engine as ensemble
import robust_decision_engine as robust
import nonlinear_complex_engine as cplx
ENGINE_VERSION='1.0.0'
def _row(name,ok,evidence=None):return {'name':name,'status':'PASS' if ok else 'FAIL','evidence':evidence}
def run(seed:int=20261005):
    rows=[]
    p=parent.run(seed);rows.append(_row('parent_v09_release_gate',p['status']=='PASS',{'status':p['status'],'rows':p['rows']}))
    e=ensemble.combine({'models':[{'name':'A','mean':[0.0],'covariance':[[1.0]],'weight':.5},{'name':'B','mean':[2.0],'covariance':[[1.0]],'weight':.5}]})
    ok=abs(e['mixture_mean'][0]-1)<1e-12 and abs(e['within_model_covariance'][0][0]-1)<1e-12 and abs(e['between_model_covariance'][0][0]-1)<1e-12
    rows.append(_row('structural_model_uncertainty',ok,{'mean':e['mixture_mean'][0],'within':e['within_model_covariance'][0][0],'between':e['between_model_covariance'][0][0]}))
    d=robust.analyze({'actions':['risky','safe'],'scenarios':['good','bad'],'loss_matrix':[[0,10],[4,4]],'weight_intervals':[[.2,.9],[.1,.8]],'nominal_weights':[.9,.1]})
    ok=d['bayes_action']=='risky' and d['minimax_expected_loss_action']=='safe'
    rows.append(_row('robust_decision_under_model_weight_uncertainty',ok,{'bayes':d['bayes_action'],'minimax':d['minimax_expected_loss_action'],'minimax_regret':d['minimax_regret_action']}))
    C=np.array([[.04,.01],[.01,.09]]);T=np.array([[2,-3],[3,2]],float);exact=T@C@T.T
    z=cplx.propagate({'variables':['z'],'values':[{'re':1,'im':2}],'covariance':C.tolist(),'expression':'(2+3j)*z','n':90000,'seed':seed})
    ok=z['status']=='PASS' and np.allclose(np.asarray(z['local_linearized_real_im_covariance']),exact,rtol=1e-6,atol=1e-8) and np.allclose(np.asarray(z['output_real_im_covariance']),exact,rtol=.025,atol=.004)
    rows.append(_row('nonlinear_complex_propagation',ok,{'mc_covariance':z['output_real_im_covariance'],'exact_linear_covariance':exact.tolist(),'invalid_fraction':z['invalid_fraction']}))
    status='PASS' if all(x['status']=='PASS' for x in rows) else 'FAIL'
    return {'engine_version':ENGINE_VERSION,'gate':'v1.0_integrated_release_gate','status':status,'seed':int(seed),'rows':rows,
            'claim_boundary':'This verifies the broad plugin-only v1.0 scope and complete v0.9 parent gate. It is not formal accreditation, external simulator/HPC qualification, or proof that the user model set is complete.'}
if __name__=='__main__':
 import json
 r=run();print(json.dumps(r,indent=2,allow_nan=False));raise SystemExit(0 if r['status']=='PASS' else 2)
