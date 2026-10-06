import gum5_full_benchmarks as g

def test_all_16_examples_have_hard_gate_and_pass():
    r=g.run_all(); assert r['hard_gate_examples']==16; assert r['status']=='PASS'
    assert [x['example'] for x in r['cases']]==list(range(1,17))
    assert all(x['checks'] and x['status']=='PASS' for x in r['cases'])

def test_claim_boundary_is_not_full_certification():
    r=g.run_all(); assert 'not formal JCGM certification' in r['claim_boundary']

def test_partial_gates_are_explicit():
    r=g.run_all(); by={x['example']:x for x in r['cases']}
    assert 'submodel' in by[8]['gate_scope'] or 'model-center' in by[8]['gate_scope']
    assert 'ANOVA' in by[12]['gate_scope']
