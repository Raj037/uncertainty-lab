import math
import numpy as np
import pytest
import constrained_nuts_engine as c


def test_positive_lognormal_parameter_recovers_simple_posterior():
    # y ~ N(theta, .2), theta lognormal with median 1 and moderate prior spread.
    y=[1.05,1.0,.95,1.1,.9,1.0]
    r=c.nuts({'parameter_specs':[{'name':'theta','kind':'lognormal','log_mean':0.0,'log_std':.5}],
              'observed':y,'observation_sd':.2,'model_expression':'theta','chains':4,'warmup':700,'draws':1500,'seed':14,
              'min_bulk_ess':200,'min_tail_ess':200})
    assert r['status']=='POSTERIOR_QUALIFIED',r['diagnostics']
    assert r['posterior_mean'][0] > 0
    assert r['posterior_mean'][0]==pytest.approx(1.0,abs=.08)
    assert sum(r['post_warmup_divergences'])==0


def test_bounded_parameter_stays_inside_bounds():
    y=[.72,.68,.70,.71,.69,.73]
    r=c.nuts({'parameter_specs':[{'name':'p','kind':'logit_normal','lower':0,'upper':1,'logit_mean':0,'logit_std':1.5}],
              'observed':y,'observation_sd':.05,'model_expression':'p','chains':4,'warmup':700,'draws':1500,'seed':7,
              'min_bulk_ess':200,'min_tail_ess':200})
    assert r['status']=='POSTERIOR_QUALIFIED'
    assert 0 < r['posterior_mean'][0] < 1
    assert r['posterior_mean'][0]==pytest.approx(.705,abs=.04)


def test_simplex_transform_preserves_sum_and_recovers_composition():
    # Two simplex components enter model as a constant a; repeated observations identify a.
    y=[.30,.31,.29,.305,.295]
    r=c.nuts({'parameter_specs':[{'kind':'simplex_logistic_normal','names':['a','b'],'latent_mean':[0.0],'latent_std':[1.5]}],
              'observed':y,'observation_sd':.03,'model_expression':'a','chains':4,'warmup':700,'draws':1500,'seed':12,
              'min_bulk_ess':200,'min_tail_ess':200})
    assert r['status']=='POSTERIOR_QUALIFIED'
    assert sum(r['posterior_mean'])==pytest.approx(1.0,abs=1e-12)
    assert r['posterior_mean'][0]==pytest.approx(.30,abs=.04)


def test_metric_adaptation_is_scale_invariant_for_log_parameterization():
    base={'parameter_specs':[{'name':'theta','kind':'lognormal','log_mean':0.0,'log_std':.4}],
          'observed':[1.0,1.05,.95,1.02,.98],'observation_sd':.08,'model_expression':'theta','chains':4,'warmup':700,'draws':1500,'seed':44,
          'min_bulk_ess':200,'min_tail_ess':200}
    a=c.nuts(base)
    # Same latent posterior, physical parameter scaled by 1e-20.
    b=c.nuts({**base,'parameter_specs':[{'name':'theta','kind':'lognormal','log_mean':math.log(1e-20),'log_std':.4}],
              'observed':[x*1e-20 for x in base['observed']],'observation_sd':.08e-20})
    assert a['status']==b['status']=='POSTERIOR_QUALIFIED'
    assert b['posterior_mean'][0]/1e-20==pytest.approx(a['posterior_mean'][0],rel=.05)
