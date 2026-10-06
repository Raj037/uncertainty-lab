#!/usr/bin/env python3
"""Integrated release gate for Uncertainty Lab v0.9.0."""
from __future__ import annotations
import math
import numpy as np
from scipy import stats
import v08_release_gate as parent
import constrained_nuts_engine as cnuts
import pbox_engine as pbox
import time_reliability_engine as trel
import boolean_system_engine as bsys
import multifidelity_engine as mf
import global_design_engine as gdesign

ENGINE_VERSION='0.9.0'
def _row(name,ok,evidence=None):return {'name':name,'status':'PASS' if ok else 'FAIL','evidence':evidence}

def run(seed:int=20261005):
    rows=[]
    p=parent.run(seed);rows.append(_row('parent_v08_release_gate',p['status']=='PASS',{'status':p['status'],'rows':p['rows']}))
    # constrained NUTS: positive parameter benchmark with strict diagnostics
    y=[1.0,1.05,.95,1.02,.98]
    n=cnuts.nuts({'parameter_specs':[{'name':'theta','kind':'lognormal','log_mean':0.0,'log_std':.4}], 'observed':y,'observation_sd':.08,'model_expression':'theta',
                  'chains':4,'warmup':700,'draws':1500,'seed':seed,'min_bulk_ess':200,'min_tail_ess':200})
    ok=n['status']=='POSTERIOR_QUALIFIED' and abs(n['posterior_mean'][0]-1.0)<.06 and sum(n['post_warmup_divergences'])==0
    rows.append(_row('constrained_nuts',ok,{'mean':n['posterior_mean'][0],'diagnostics':n['diagnostics']['status'],'divergences':n['post_warmup_divergences']}))
    # p-box exact scalar CDF
    pb=pbox.cdf_bounds({'kind':'normal','mean':[-1,1],'std':[1,1]},0)
    ok=abs(pb['cdf_lower']-float(stats.norm.cdf(-1)))<1e-8 and abs(pb['cdf_upper']-float(stats.norm.cdf(1)))<1e-8
    rows.append(_row('epistemic_pbox',ok,{'lower':pb['cdf_lower'],'upper':pb['cdf_upper']}))
    # continuous Brownian first passage against exact reflection result
    br=trel.brownian({'x0':0,'drift':0,'sigma':1,'barrier':1,'horizon':1,'steps':80,'n':70000,'seed':seed});exact=2*float(stats.norm.sf(1))
    ok=abs(br['failure_probability']-exact)<5*br['standard_error']+.004
    rows.append(_row('time_dependent_reliability',ok,{'estimate':br['failure_probability'],'exact':exact,'se':br['standard_error']}))
    # general boolean system with shared common-cause input
    bs=bsys.evaluate({'variables':['x'],'joint_model':{'kind':'multivariate_normal','mean':[0],'covariance':[[1]]},'n':120000,'seed':seed,
                      'components':{'A':'1-x','B':'1.2-x','C':'2-x'},'system_event':'A or (B and C)'})
    exact=float(stats.norm.sf(1));ok=abs(bs['system_failure_probability']-exact) <= 5*bs['standard_error']+5e-4 and ['A'] in bs['minimal_cut_sets']
    rows.append(_row('boolean_system_reliability',ok,{'pf':bs['system_failure_probability'],'exact':exact,'se':bs['standard_error'],'cut_sets':bs['minimal_cut_sets']}))
    # multi-fidelity control variate
    rng=np.random.default_rng(seed);x=rng.normal(size=180);H=x+.2*rng.normal(size=180);L=x+.35*rng.normal(size=180);E=rng.normal(size=4000)+.35*rng.normal(size=4000)
    cv=mf.control_variate({'high_values':H,'low_paired_values':L,'low_extra_values':E});ok=cv['standard_error_estimate']<cv['naive_high_only_standard_error'] and abs(cv['estimate'])<.12
    rows.append(_row('multifidelity_uq',ok,{'estimate':cv['estimate'],'se':cv['standard_error_estimate'],'naive_se':cv['naive_high_only_standard_error'],'factor':cv['estimated_se_reduction_factor']}))
    # global and nonlinear experiment design
    X=rng.normal(size=(900,2));gd=gdesign.nonlinear_global_batch({'parameter_names':['a','b'],'parameter_samples':X.tolist(),'batch_size':2,'outer_samples':180,'replicates':3,'seed':seed,
        'candidates':[{'name':'a','model_expression':'a','noise_std':.4},{'name':'b','model_expression':'b','noise_std':.4},{'name':'redundant_a','model_expression':'a','noise_std':.8}]})
    ok=set(gd['best_batch']['candidates'])=={'a','b'}
    rows.append(_row('global_bayesian_experimental_design',ok,{'best_batch':gd['best_batch']}))
    status='PASS' if all(x['status']=='PASS' for x in rows) else 'FAIL'
    return {'engine_version':ENGINE_VERSION,'gate':'v0.9_integrated_release_gate','status':status,'seed':int(seed),'rows':rows,
            'claim_boundary':'This verifies the encoded v0.9 plugin-only scope plus the complete v0.8 parent release gate; it is not formal accreditation or proof that all epistemic/model uncertainty has been captured.'}
if __name__=='__main__':
    import json
    z=run();print(json.dumps(z,indent=2,allow_nan=False));raise SystemExit(0 if z['status']=='PASS' else 2)
