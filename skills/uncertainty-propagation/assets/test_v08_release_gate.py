import v08_release_gate as g
def test_v08_gate():
    r=g.run(20261005);assert r['status']=='PASS',r;assert len(r['rows'])==7;assert all(x['status']=='PASS' for x in r['rows'])
