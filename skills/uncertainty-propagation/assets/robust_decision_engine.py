#!/usr/bin/env python3
"""Robust decisions under uncertain model/scenario weights for Uncertainty Lab v1.0."""
from __future__ import annotations
from typing import Any, Mapping
import numpy as np
from scipy.optimize import linprog
ENGINE_VERSION='1.0.0'
class InputError(ValueError):pass


def _problem(payload):
    actions=list(payload.get('actions') or []);scenarios=list(payload.get('scenarios') or []);L=np.asarray(payload.get('loss_matrix'),dtype=float)
    if not actions or not scenarios or len(set(actions))!=len(actions) or len(set(scenarios))!=len(scenarios) or L.shape!=(len(actions),len(scenarios)) or not np.isfinite(L).all():raise InputError('actions/scenarios/loss_matrix mismatch')
    ints=np.asarray(payload.get('weight_intervals'),dtype=float)
    if ints.shape!=(len(scenarios),2) or np.any(ints<0) or np.any(ints[:,0]>ints[:,1]) or ints[:,0].sum()>1+1e-12 or ints[:,1].sum()<1-1e-12:raise InputError('invalid/infeasible weight_intervals')
    nominal=payload.get('nominal_weights')
    if nominal is None: nominal=np.mean(ints,axis=1); nominal=nominal/nominal.sum()
    else:
        nominal=np.asarray(nominal,dtype=float)
        if nominal.shape!=(len(scenarios),) or np.any(nominal<0) or nominal.sum()<=0:raise InputError('nominal_weights invalid')
        nominal=nominal/nominal.sum()
        if np.any(nominal<ints[:,0]-1e-12) or np.any(nominal>ints[:,1]+1e-12):raise InputError('nominal_weights outside intervals')
    return actions,scenarios,L,ints,nominal

def _extreme(v,ints,maximize=False):
    c=-np.asarray(v) if maximize else np.asarray(v);res=linprog(c,A_eq=np.ones((1,len(v))),b_eq=[1.0],bounds=[tuple(x) for x in ints],method='highs')
    if not res.success:raise InputError('robust optimization failed')
    val=float(np.asarray(v)@res.x);return val,res.x.tolist()

def analyze(payload:Mapping[str,Any])->dict[str,Any]:
    actions,scenarios,L,ints,w0=_problem(payload);best_by_scenario=np.min(L,axis=0);reg=L-best_by_scenario
    rows=[]
    for i,a in enumerate(actions):
        lo,wlo=_extreme(L[i],ints,False);hi,whi=_extreme(L[i],ints,True);worst_reg,wreg=_extreme(reg[i],ints,True)
        rows.append({'action':a,'nominal_expected_loss':float(L[i]@w0),'best_case_expected_loss':lo,'worst_case_expected_loss':hi,'worst_case_regret':worst_reg,'worst_loss_weights':whi,'worst_regret_weights':wreg})
    bayes=min(rows,key=lambda r:r['nominal_expected_loss']);minimax=min(rows,key=lambda r:r['worst_case_expected_loss']);mmr=min(rows,key=lambda r:r['worst_case_regret'])
    return {'engine_version':ENGINE_VERSION,'kind':'robust_decision_weight_polytope','status':'PASS','actions':actions,'scenarios':scenarios,'nominal_weights':w0.tolist(),'analysis':rows,'bayes_action':bayes['action'],'minimax_expected_loss_action':minimax['action'],'minimax_regret_action':mmr['action'],
            'claim_boundary':'Robustness is exact only over the declared scenario set, loss matrix, and model-weight polytope; missing scenarios and misspecified losses remain outside the guarantee.'}

def robust_utility_design(payload:Mapping[str,Any])->dict[str,Any]:
    names=list(payload.get('candidates') or []);sc=list(payload.get('scenarios') or []);U=np.asarray(payload.get('utility_matrix'),dtype=float);ints=np.asarray(payload.get('weight_intervals'),dtype=float)
    if not names or not sc or U.shape!=(len(names),len(sc)):raise InputError('candidates/scenarios/utility_matrix mismatch')
    if ints.shape!=(len(sc),2) or ints[:,0].sum()>1+1e-12 or ints[:,1].sum()<1-1e-12:raise InputError('invalid weight intervals')
    rows=[]
    for i,n in enumerate(names):
        worst,w=_extreme(U[i],ints,False);best,_=_extreme(U[i],ints,True);rows.append({'candidate':n,'worst_case_utility':worst,'best_case_utility':best,'worst_case_weights':w})
    rows.sort(key=lambda r:r['worst_case_utility'],reverse=True)
    return {'engine_version':ENGINE_VERSION,'kind':'maximin_robust_candidate_design','status':'PASS','ranked_candidates':rows,'best_candidate':rows[0],
            'claim_boundary':'Maximin design is with respect to the declared finite scenario-weight polytope and utility matrix, not unknown omitted models.'}
