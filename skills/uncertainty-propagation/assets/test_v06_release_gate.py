import v06_release_gate as gate

def test_v06_integrated_release_gate_passes():
    r=gate.run(20261005)
    assert r['status']=='PASS', r
    assert all(x['status']=='PASS' for x in r['rows'])
