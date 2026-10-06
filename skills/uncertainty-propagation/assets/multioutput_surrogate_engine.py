#!/usr/bin/env python3
"""Qualified multi-output surrogate models for Uncertainty Lab v0.8.

PCE uses a shared orthonormal basis and therefore returns analytic cross-output
covariance induced by common inputs. GP uses independently fitted output GPs;
input-driven cross-output covariance is retained through joint prediction over
common input samples, while cross-output GP residual covariance is not claimed.
"""
from __future__ import annotations
import math
from typing import Any,Mapping
import numpy as np
import surrogate_engine as sg
import joint_distribution_engine as joint

ENGINE_VERSION='0.8.0'
MAX_OUTPUTS=16
class InputError(ValueError):pass

def _xy(payload):
    X=np.asarray(payload.get('X'),dtype=float);Y=np.asarray(payload.get('Y'),dtype=float)
    if X.ndim!=2 or Y.ndim!=2 or X.shape[0]!=Y.shape[0] or X.shape[0]<10 or not 1<=Y.shape[1]<=MAX_OUTPUTS or not np.isfinite(X).all() or not np.isfinite(Y).all():raise InputError('X/Y must be aligned finite nxd / nxm data with >=10 rows')
    return X,Y

def pce(payload:Mapping[str,Any])->dict[str,Any]:
    X,Y=_xy(payload);marg=list(payload.get('marginals') or []);degree=int(payload.get('total_degree',2));models=[];runs=[];statuses=[]
    for j in range(Y.shape[1]):
        q={**payload,'X':X.tolist(),'y':Y[:,j].tolist()};r=sg.polynomial_chaos(q);models.append(r);statuses.append(r['status']);runs.append({'output':j,'status':r['status'],'validation_metrics':r['validation_metrics']})
    if not all(s=='SURROGATE_QUALIFIED' for s in statuses):
        return {'engine_version':ENGINE_VERSION,'kind':'multioutput_pce','status':'SURROGATE_REJECTED','outputs':runs}
    # Refit all outputs on all data using the identical orthonormal basis.
    fits=[sg._pce_fit_raw(X,Y[:,j],marg,degree) for j in range(Y.shape[1])];C=np.column_stack([f['coef'] for f in fits]);mean=C[0];cov=C[1:].T@C[1:]
    return {'engine_version':ENGINE_VERSION,'kind':'multioutput_pce','status':'SURROGATE_QUALIFIED','output_dimension':Y.shape[1],'output_mean':mean.tolist(),'output_covariance':cov.tolist(),'output_std':np.sqrt(np.clip(np.diag(cov),0,None)).tolist(),'outputs':runs,
            'claim_boundary':'Cross-output covariance is analytic under the shared orthonormal PCE basis and declared independent input distribution.'}

def gp_propagate(payload:Mapping[str,Any])->dict[str,Any]:
    X,Y=_xy(payload);dist=dict(payload.get('input_distribution') or {});n=int(payload.get('n',30000));seed=int(payload.get('seed',1729));dist['n']=n;dist['seed']=seed
    validations=[];fits=[]
    for j in range(Y.shape[1]):
        q={**payload,'X':X.tolist(),'y':Y[:,j].tolist()};vr=sg.gaussian_process(q);validations.append({'output':j,'status':vr['status'],'validation_metrics':vr['validation_metrics']})
        if vr['status']!='SURROGATE_QUALIFIED':return {'engine_version':ENGINE_VERSION,'kind':'multioutput_independent_gp','status':'SURROGATE_REJECTED','outputs':validations}
        fits.append(sg._gp_fit_raw(X,Y[:,j],q))
    names,S,_=joint.generate_inputs(dist);means=[];pvars=[]
    for f in fits:
        m,v=sg._gp_predict(f,S);means.append(m);pvars.append(v)
    M=np.column_stack(means);PV=np.column_stack(pvars);cov=np.atleast_2d(np.cov(M,rowvar=False,ddof=1))
    return {'engine_version':ENGINE_VERSION,'kind':'multioutput_independent_gp','status':'SURROGATE_QUALIFIED','input_variable_order':names,'n_propagation':len(S),'output_mean':M.mean(0).tolist(),'input_driven_output_covariance':cov.tolist(),'mean_marginal_GP_predictive_variance':PV.mean(0).tolist(),'outputs':validations,
            'claim_boundary':'Outputs are fitted by independent GPs. Shared-input propagation preserves input-driven cross-output covariance, but cross-output GP residual/predictive covariance is not modeled or claimed.'}
