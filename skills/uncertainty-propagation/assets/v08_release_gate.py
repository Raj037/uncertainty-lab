#!/usr/bin/env python3
"""Integrated release gate for Uncertainty Lab v0.8.0."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from scipy import stats
import v07_release_gate as parent
import non_gaussian_reliability_engine as ngr
import hmc_nuts_engine as hmc
import calibration_engine as cal
import discrepancy_inference_engine as disc
import active_learning_reliability_engine as al
import multioutput_surrogate_engine as mos
import system_reliability_engine as system
import experimental_design_engine as design

ENGINE_VERSION='0.8.0'
def _row(name,ok,evidence=None):return {'name':name,'status':'PASS' if ok else 'FAIL','evidence':evidence}

def run(seed:int=20261005):
    rows=[]
    p=parent.run(seed);rows.append(_row('parent_v07_release_gate',p['status']=='PASS',{'status':p['status'],'rows':p['rows']}))
    # Non-Gaussian reliability: exact lognormal 5% tail by two different routes.
    q=float(stats.lognorm.ppf(.95,s=.4,scale=math.exp(.2)));base={'variables':['x'],'joint_model':{'kind':'gaussian_copula','copula_correlation':[[1.0]],'marginals':[{'kind':'lognormal','log_mean':.2,'log_std':.4}]},'limit_state':f'{q}-x','seed':seed}
    rq=ngr.randomized_qmc({**base,'power':13,'replicates':6});rd=ngr.directional({**base,'directions':700,'replicates':4})
    ok=abs(rq['failure_probability']-.05)<.003 and abs(rd['failure_probability']-.05)<.006
    rows.append(_row('non_gaussian_reliability',ok,{'qmc':rq['failure_probability'],'qmc_se':rq['standard_error_across_scrambles'],'directional':rd['failure_probability'],'directional_se':rd['standard_error_across_direction_replicates']}))
    # NUTS against exact posterior.
    y=[1.0,1.2,.8,1.1,.9,1.05,.95,1.1];exact=cal.linear_gaussian({'observed':y,'design_matrix':np.ones((len(y),1)).tolist(),'prior_mean':[0.0],'prior_covariance':[[1.0]],'observation_sd':.5})
    npay={'parameter_names':['theta'],'priors':{'theta':{'kind':'normal','mean':0,'std':1}},'observed':y,'observation_sd':.5,'model_expression':'theta','chains':4,'warmup':450,'draws':900,'seed':seed,'min_bulk_ess':150,'min_tail_ess':150,'max_treedepth':7}
    nr=hmc.nuts(npay);ok=nr['status']=='POSTERIOR_QUALIFIED' and abs(nr['posterior_mean'][0]-exact['posterior_mean'][0])<.05 and abs(nr['posterior_sd'][0]-exact['posterior_sd'][0])<.04 and sum(nr['post_warmup_divergent_iterations'])==0
    rows.append(_row('nuts_calibration',ok,{'mean':nr['posterior_mean'][0],'exact_mean':exact['posterior_mean'][0],'sd':nr['posterior_sd'][0],'exact_sd':exact['posterior_sd'][0],'diagnostics':nr['diagnostics']['status'],'divergences':nr['post_warmup_divergent_iterations']}))
    # Joint discrepancy: white discrepancy identifiable enough; aligned kernel is flagged.
    rng=np.random.default_rng(seed);n=32;theta=1.1;tau=.25;obs=.1;yy=theta+rng.normal(scale=math.sqrt(obs*obs+tau*tau),size=n)
    d1=disc.infer_linear_gaussian({'observed':yy.tolist(),'design_matrix':np.ones((n,1)).tolist(),'prior_mean':[0.0],'prior_covariance':[[4.0]],'observation_sd':obs,'discrepancy_kernel':np.eye(n).tolist(),'tau_prior_scale':.5})
    d2=disc.infer_linear_gaussian({'observed':np.ones(n).tolist(),'design_matrix':np.ones((n,1)).tolist(),'prior_mean':[0.0],'prior_covariance':[[1.0]],'observation_sd':.1,'discrepancy_kernel':np.ones((n,n)).tolist(),'tau_prior_scale':.5})
    ok=d1['status']=='POSTERIOR_QUALIFIED' and abs(d1['posterior_parameter_mean'][0]-theta)<.15 and abs(d1['tau_posterior_mean']-tau)<.15 and d2['status'].startswith('POSTERIOR_PROVISIONAL') and d2['parameter_discrepancy_confounding_fraction']>.99
    rows.append(_row('joint_discrepancy_identifiability',ok,{'theta':d1['posterior_parameter_mean'][0],'tau':d1['tau_posterior_mean'],'confounded_fraction':d2['parameter_discrepancy_confounding_fraction'],'confounded_status':d2['status']}))
    # Active learning on known 3-sigma limit state.
    X=np.array([[-3.],[-2.],[-1.],[0.],[1.],[2.],[4.]]);g=3-X[:,0]
    ar=al.active_learn_expression({'X':X.tolist(),'g':g.tolist(),'variables':['x'],'input_distribution':{'variables':['x'],'joint_model':{'kind':'multivariate_normal','mean':[0.0],'covariance':[[1.0]]}},'candidate_n':4096,'candidate_seed':seed,'gp_noise_std':1e-8,'oracle_expression':'3-x','evaluation_budget':6,'u_stop_threshold':2.0,'max_classification_interval_width':.003})
    ok=abs(ar['failure_probability_surrogate_mean_classifier']-float(stats.norm.sf(3)))<.0025 and ar['classification_probability_lower']<=ar['failure_probability_surrogate_mean_classifier']<=ar['classification_probability_upper']
    rows.append(_row('active_learning_reliability',ok,{'status':ar['status'],'pf':ar['failure_probability_surrogate_mean_classifier'],'interval':[ar['classification_probability_lower'],ar['classification_probability_upper']],'new_evaluations':ar['new_evaluations']}))
    # Multi-output PCE exact covariance + system reliability direct shared sampling.
    X2=rng.normal(size=(220,2));Y=np.column_stack([X2[:,0]+X2[:,1],2*X2[:,0]])
    mr=mos.pce({'X':X2.tolist(),'Y':Y.tolist(),'marginals':[{'kind':'normal','mean':0,'std':1}]*2,'total_degree':1,'validation_fraction':.2,'validation_repeats':2,'qualification':{'max_nrmse':1e-9,'min_r2':.99999999}})
    sr=system.randomized_qmc({'variables':['x','y'],'joint_model':{'kind':'multivariate_normal','mean':[0,0],'covariance':[[1,0],[0,1]]},'limit_states':['2-x','2-y'],'system':{'kind':'series'},'power':13,'replicates':6,'seed':seed});pp=float(stats.norm.sf(2));sex=1-(1-pp)**2
    ok=mr['status']=='SURROGATE_QUALIFIED' and np.allclose(np.asarray(mr['output_covariance']),[[2,2],[2,4]],atol=1e-8) and abs(sr['failure_probability']-sex)<.0015
    rows.append(_row('multioutput_and_system_reliability',ok,{'pce_covariance':mr.get('output_covariance'),'series_pf':sr['failure_probability'],'series_exact':sex}))
    # Experimental design: high-variance direction for information, target direction for c-optimal objective.
    dp={'parameter_mean':[0,0],'parameter_covariance':[[4,0],[0,1]],'candidates':[{'name':'a','sensitivity':[1,0],'noise_std':.1,'cost':1},{'name':'b','sensitivity':[0,1],'noise_std':.1,'cost':1},{'name':'mixed','sensitivity':[1,1],'noise_std':.5,'cost':3}]}
    info=design.score_candidates({**dp,'objective':'information_gain'});target=design.score_candidates({**dp,'objective':'target_variance_reduction','target_sensitivity':[0,1]});batch=design.greedy_batch({**dp,'objective':'information_gain','batch_size':2})
    ok=info['best_candidate']['name']=='a' and target['best_candidate']['name']=='b' and len(set(batch['selected_candidates']))==2 and batch['total_information_gain_nats']>0
    rows.append(_row('experimental_design_value_of_information',ok,{'information_best':info['best_candidate']['name'],'target_best':target['best_candidate']['name'],'batch':batch['selected_candidates'],'batch_information_gain':batch['total_information_gain_nats']}))
    status='PASS' if all(x['status']=='PASS' for x in rows) else 'FAIL'
    return {'engine_version':ENGINE_VERSION,'gate':'v0.8_integrated_release_gate','status':status,'seed':int(seed),'rows':rows,'claim_boundary':'This gate verifies the encoded v0.8 scope and complete v0.7 parent gate. It is not formal reliability/metrology accreditation, proof of model completeness, or proof of globally optimal experiment design.'}
if __name__=='__main__':
    import json
    z=run();print(json.dumps(z,indent=2,allow_nan=False));raise SystemExit(0 if z['status']=='PASS' else 2)
