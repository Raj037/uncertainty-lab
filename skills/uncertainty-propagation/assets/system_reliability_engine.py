#!/usr/bin/env python3
"""Series/parallel/k-out-of-n system reliability under explicit joint inputs (v0.8)."""
from __future__ import annotations
import math
from typing import Any,Mapping
import numpy as np
from scipy.stats import qmc
import joint_distribution_engine as joint
import non_gaussian_reliability_engine as ngr

ENGINE_VERSION='0.8.0'
MAX_EVENTS=32
class InputError(ValueError):pass

def randomized_qmc(payload:Mapping[str,Any])->dict[str,Any]:
    setup=ngr._joint_setup(payload);names,kind,qdim,_,_,_=setup;events=list(payload.get('limit_states') or [])
    if not 1<=len(events)<=MAX_EVENTS or any(not isinstance(e,str) or not e.strip() for e in events):raise InputError(f'limit_states must contain 1..{MAX_EVENTS} expressions')
    nodes=[joint._compile(e) for e in events];sys=dict(payload.get('system') or {});sk=str(sys.get('kind','series')).lower();kreq=int(sys.get('k',1))
    if sk not in {'series','parallel','k_out_of_n'}:raise InputError('system.kind must be series, parallel, or k_out_of_n')
    if sk=='k_out_of_n' and not 1<=kreq<=len(events):raise InputError('system.k out of range')
    power=int(payload.get('power',14));reps=int(payload.get('replicates',8));seed=int(payload.get('seed',1729))
    if power<8 or power>21 or reps<2 or reps>64:raise InputError('invalid QMC settings')
    sysps=[];ind=[];corrs=[]
    for r in range(reps):
        U=qmc.Sobol(qdim,scramble=True,seed=seed+r*104729).random_base2(power);X=ngr._from_unit(U,setup);I=np.column_stack([ngr._eval(n,names,X)<=0 for n in nodes]);count=I.sum(1)
        if sk=='series':S=count>=1
        elif sk=='parallel':S=count==len(events)
        else:S=count>=kreq
        sysps.append(float(S.mean()));ind.append(I.mean(0));corrs.append(np.corrcoef(I.astype(float),rowvar=False) if len(events)>1 else np.array([[1.]]))
    a=np.asarray(sysps);ip=np.vstack(ind)
    return {'engine_version':ENGINE_VERSION,'kind':'randomized_qmc_system_reliability','status':'PASS','system':{'kind':sk,**({'k':kreq} if sk=='k_out_of_n' else {})},'failure_probability':float(a.mean()),'standard_error_across_scrambles':float(a.std(ddof=1)/math.sqrt(reps)),'replicate_probabilities':a.tolist(),'individual_event_probabilities':ip.mean(0).tolist(),'mean_event_indicator_correlation':np.mean(np.stack(corrs),axis=0).tolist(),'events':events,'samples_per_replicate':2**power,'replicates':reps,'joint_model_kind':kind,'claim_boundary':'System probability is evaluated directly on shared joint-input samples, preserving common-cause dependence. It is not computed from independence products of component probabilities.'}
