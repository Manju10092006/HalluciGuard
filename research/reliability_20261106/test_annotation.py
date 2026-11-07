import pytest
from annotation_agreement import compare

def row(id,label):return {'item_id':id,'support_label':label}

def test_blank_labels_are_pending_not_agreement():
    assert compare([row('a','')],[row('a','')])['cohen_kappa'] is None
    assert compare([row('a','')],[row('a','')])['completed_pairs']==0

def test_disagreement_and_missing_labels_remain_separate():
    result=compare([row('a','supported'),row('b','contradicted'),row('c','')],
                   [row('a','supported'),row('b','supported'),row('c','supported')])
    assert result['completed_pairs']==2 and result['agreement']==0.5
    assert result['disagreements']==['b']

def test_duplicate_annotation_ids_fail():
    with pytest.raises(ValueError):compare([row('a',''),row('a','')],[])
