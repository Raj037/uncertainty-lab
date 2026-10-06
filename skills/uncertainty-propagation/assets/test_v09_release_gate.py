import v09_release_gate as g

def test_v09_release_gate_passes():
    r=g.run(20261005)
    assert r['status']=='PASS',r
    assert len(r['rows'])==7
    assert all(x['status']=='PASS' for x in r['rows'])
