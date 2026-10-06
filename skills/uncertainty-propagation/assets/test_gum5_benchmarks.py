import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import gum5_benchmarks as g

def test_six_official_gates():
    assert g.case_3_glucose()['status']=='PASS'
    assert g.case_6_2()['status']=='PASS'
    assert g.case_6_3(n=500000)['status']=='PASS'
    assert g.case_6_4(n=500000)['status']=='PASS'
    assert g.case_7()['status']=='PASS'
    assert g.case_14()['status']=='PASS'
def test_document_claim_boundary():
    assert len(g.DOCUMENT_COVERAGE)==16
    assert sum(v['gate']=='hard_gate' for v in g.DOCUMENT_COVERAGE.values())==4
    r=g.run_release_gate(n=500000);assert r['status']=='PASS';assert r['hard_gate_count']==6;assert 'not a claim' in r['claim_boundary']
def test_official_doi_is_current_final_2026_doi():assert g.SOURCE['doi']=='10.59161/YNLY8209'
