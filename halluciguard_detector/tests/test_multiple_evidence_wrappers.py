from halluciguard_detector.structured_evidence import normalize_evidence


def test_multiple_wrapper_collections_preserve_every_source_without_gluing():
    records, trace = normalize_evidence({
        "source_id": "parent", "rating": 4,
        "evidence": [{"source_id": "A", "text": "Shop closes Monday."}],
        "documents": [{"source_id": "B", "text": "Shop opens Tuesday."}],
        "passages": None,
    })
    assert records == ["rating: 4", "Shop closes Monday.", "Shop opens Tuesday."]
    assert trace["source_ids"] == ["parent", "A", "B"]
    assert trace["normalized_count"] == 3
    assert not trace["degraded"]


def test_multiple_wrappers_still_report_bounded_document_truncation():
    records, trace = normalize_evidence({
        "evidence": ["first"] * 64, "documents": ["must not silently vanish"],
    })
    assert len(records) == 64
    assert trace["document_truncated"] and trace["degraded"]
