import numpy as np
import pytest
import nonlinear_complex_engine as n


def test_linear_complex_matches_existing_exact_engine():
    p={'variables':['z'],'values':[{'re':1,'im':2}],'covariance':[[.04,.01],[.01,.09]],'expression':'(2+3j)*z','n':120000,'seed':3}
    r=n.propagate(p)
    T=np.array([[2,-3],[3,2]],dtype=float); ex_cov=T@np.array(p['covariance'])@T.T
    assert r['status']=='PASS'
    assert np.allclose(np.array(r['local_linearized_real_im_covariance']),ex_cov,rtol=1e-6,atol=1e-8)
    assert np.allclose(np.array(r['output_real_im_covariance']),ex_cov,rtol=.02,atol=.003)


def test_nonlinear_square_circular_noise_mean_near_nominal_square():
    r=n.propagate({'variables':['z'],'values':[{'re':1,'im':1}],'covariance':[[.02,0],[0,.02]],'expression':'z**2','n':100000,'seed':5})
    m=r['mean'][0]
    assert m['re']==pytest.approx(0,abs=.015)
    assert m['im']==pytest.approx(2,abs=.02)


def test_phase_resultant_is_bounded():
    r=n.propagate({'variables':['z'],'values':[{'re':1,'im':0}],'covariance':[[.05,0],[0,.05]],'expression':'exp(z)','n':20000,'seed':9})
    assert 0<=r['phase'][0]['resultant_length']<=1


def test_unsafe_attribute_syntax_blocked_at_evaluation():
    with pytest.raises(n.InputError): n.propagate({'variables':['z'],'values':[1],'covariance':[[1,0],[0,1]],'expression':'z.real','n':1000})


def test_deterministic_seed():
    p={'variables':['z'],'values':[{'re':.5,'im':.2}],'covariance':[[.02,0],[0,.03]],'expression':'z*z+exp(z)','n':5000,'seed':77}
    assert n.propagate(p)==n.propagate(p)
