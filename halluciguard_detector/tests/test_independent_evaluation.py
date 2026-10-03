"""Synthetic fixtures test software only: no real Detector performance claims.

No Verifier service is invoked. Existing orchestration stub tests are separately
reported as MOCKED_VERIFIER_PATH, never LIVE_VERIFIER_PASS.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from halluciguard_detector.calibration import fit_temperature
from halluciguard_detector.data_roles import RoleLabels
from halluciguard_detector.evaluation_data import (GateBlocked, LABELS, audit_splits, canonical, file_hash,
    grouped_split, labels_for_role, lock_splits, normalize, read_rows, require_isolation, verify_bundle,
    verify_checkpoint_exposure)
from halluciguard_detector.independent_evaluation import bootstrap_intervals, evaluate_locked, main, metric_report
from halluciguard_detector.training import best_threshold, train


def row(i, source=None, label=None):
    label = LABELS[i % 3] if label is None else label
    return {"id": f"row-{i}", "source_id": source or f"source-{i}", "response_id": f"response-{i}",
            "claim": hashlib.sha256(f"claim-{i}".encode()).hexdigest(),
            "evidence": hashlib.sha256(f"evidence-{i}".encode()).hexdigest(),
            "label": label, "label_id": LABELS.index(label)}


def splits():
    return {role: [row(i) for i in range(start, start + 6)]
            for role, start in zip(('training', 'development', 'locked_test'), (0, 6, 12))}


def locked(tmp_path, supplied=None, exposure=None, kind='synthetic_fixture'):
    root = tmp_path / 'bundle'
    h = lock_splits(supplied or splits(), root, dataset_name='SYNTHETIC_FRAMEWORK_FIXTURE', dataset_version='1',
                    git_sha='fixture-source-sha', created_at='2026-10-03T00:00:00Z', seed=42,
                    exposure_record=exposure, provenance_kind=kind)
    return root, h


def checkpoint(tmp_path):
    root = tmp_path / 'checkpoint'
    root.mkdir()
    for name, content in {'model.safetensors': 'synthetic-placeholder-not-a-model',
        'tokenizer.json': '{}', 'tokenizer_config.json': '{}',
        'config.json': json.dumps({'id2label': {str(i): label for i, label in enumerate(LABELS)}}),
        'calibration.json': json.dumps({'temperature': 1., 'contradiction_threshold': .46,
                                      'verification_risk_threshold': .625, 'max_length': 256})}.items():
        (root / name).write_text(content)
    return root


def exposure(tmp_path, model, history=None):
    history = history or {k: v for k, v in splits().items() if k != 'locked_test'}
    data = {'coverage': 'complete', 'attested_by': 'SYNTHETIC_TEST_FIXTURE',
            'checkpoint_sha256': file_hash(model / 'model.safetensors'),
            'calibration_sha256': file_hash(model / 'calibration.json')}
    for role, rows in history.items():
        path = tmp_path / f'historical-{role}.jsonl'
        path.write_bytes(b''.join(canonical(r) + b'\n' for r in rows))
        data[role + '_artifacts'] = [{'path': str(path), 'sha256': file_hash(path)}]
    path = tmp_path / 'exposure.json'
    path.write_bytes(canonical(data))
    return path


def test_normalization_is_unicode_deterministic_and_preserves_negation():
    assert normalize(' Ａlpha\tBETA \n') == 'alpha beta'
    assert normalize('not 1981.') != normalize('1981.')


@pytest.mark.parametrize('field', ['claim', 'evidence'])
def test_exact_normalized_duplicates_are_detected(field):
    a, b = row(0), row(1)
    a[field], b[field] = ' Ａlpha   BETA', 'alpha beta'
    audit = audit_splits({'training': [a], 'locked_test': [b]})
    assert audit['exact'][field]['cross_split_groups'] == 1
    with pytest.raises(GateBlocked, match='overlap'):
        require_isolation(audit)


def test_pair_duplicate_and_repeated_response_counts_are_explicit():
    a, b = row(0), row(1)
    b['claim'], b['evidence'], b['response_id'] = a['claim'], a['evidence'], a['response_id']
    audit = audit_splits({'training': [a, b]})
    assert audit['exact']['pair']['extra_rows'] == 1
    assert audit['exact']['response_group']['extra_rows'] == 1
    assert audit['splits']['training']['exclusion_count'] == 0


@pytest.mark.parametrize('field', ['id', 'source_id', 'response_id', 'document_id'])
def test_each_cross_split_identity_is_rejected(field):
    values = splits()
    values['training'][0][field] = 'shared'
    values['locked_test'][0][field] = 'shared'
    with pytest.raises(GateBlocked, match='overlap'):
        require_isolation(audit_splits(values))


def test_train_dev_overlap_is_not_excused_by_clean_test(tmp_path):
    values = splits()
    values['development'][0]['source_id'] = values['training'][0]['source_id']
    with pytest.raises(GateBlocked, match='overlap'):
        locked(tmp_path, values)
    assert not (tmp_path / 'bundle').exists()


def test_near_duplicates_measured_without_removing_rows():
    a, b = row(0), row(1)
    a['claim'] = 'The measured distance between the two research stations was exactly 123 kilometres.'
    b['claim'] = a['claim'].replace('123', '124')
    a['evidence'] = 'The expedition report gives a distance of 123 kilometres between the two research stations.'
    b['evidence'] = a['evidence'].replace('123', '124')
    result = audit_splits({'training': [a], 'locked_test': [b]}, near_threshold=.8)
    assert result['near']['pair_count'] == result['near']['cross_split_pair_count'] == 1
    assert result['near']['pairs'][0]['claim_similarity'] >= .8
    assert result['splits']['locked_test']['total_rows'] == 1
    assert result['splits']['locked_test']['exclusions'] == []


def test_near_candidate_budget_is_a_blocker_not_a_partial_pass():
    rows = [row(i) for i in range(4)]
    for r in rows:
        r['claim'] = 'same candidate claim'
    with pytest.raises(GateBlocked, match='audit incomplete'):
        audit_splits({'pool': rows}, max_candidates=1)


def test_grouped_regeneration_and_label_counts_are_stable():
    rows = [row(i, source=f'source-{i // 2}') for i in range(30)]
    result = grouped_split(rows, seed=18)
    assert result == grouped_split(list(reversed(rows)), seed=18)
    assert sum(len(x) for x in result.values()) == len(rows)
    assert sorted(r['id'] for rs in result.values() for r in rs) == sorted(r['id'] for r in rows)
    assert sorted(r['label'] for rs in result.values() for r in rs) == sorted(r['label'] for r in rows)
    require_isolation(audit_splits(result))


def test_duplicate_content_components_move_together():
    rows = [row(i) for i in range(12)]
    rows[1]['claim'] = rows[0]['claim']
    result = grouped_split(rows)
    roles = {r['id']: role for role, rs in result.items() for r in rs}
    assert roles['row-0'] == roles['row-1']


def test_insufficient_source_components_are_not_forced():
    with pytest.raises(GateBlocked, match='fewer than three'):
        grouped_split([row(i, source='only-source') for i in range(8)])


def test_manifest_hashes_are_stable_and_full_class_counts_preserved(tmp_path):
    a, ha = locked(tmp_path / 'a')
    b, hb = locked(tmp_path / 'b')
    assert ha == hb
    manifest, rows = verify_bundle(a, ha)
    assert manifest['audit']['splits']['locked_test']['labels'] == {k: 2 for k in LABELS}
    assert all(r['data_role'] == 'locked_test' for r in rows['locked_test'])
    with pytest.raises(FileExistsError):
        lock_splits(splits(), a, dataset_name='x', dataset_version='1', git_sha='sha')


@pytest.mark.parametrize('name', ['locked_test.jsonl', 'locked_test.manifest.json', 'gate-manifest.json'])
def test_locked_mutation_is_detected(tmp_path, name):
    root, h = locked(tmp_path)
    with (root / name).open('ab') as f:
        f.write(b' ')
    with pytest.raises(GateBlocked, match='mutat|mismatch'):
        verify_bundle(root, h)


def test_label_mismatch_is_rejected_not_remapped(tmp_path):
    values = splits()
    values['locked_test'][0]['label_id'] = 2
    with pytest.raises(GateBlocked, match='label/count'):
        locked(tmp_path, values)


def test_unknown_labels_block():
    a = row(0)
    a['label'] = 'NEI'
    with pytest.raises(GateBlocked, match='label/count'):
        audit_splits({'pool': [a]})


def test_ragtruth_response_id_is_derived_only_from_span_id():
    a = row(0)
    a['id'] = 'official-response:12-48#0'
    del a['response_id']
    result = audit_splits({'pool': [a]})
    assert result['splits']['pool']['unique_response_groups'] == 1


def test_padded_source_ids_cannot_evade_group_isolation():
    a = row(0)
    a['source_id'] = ' source-0 '
    with pytest.raises(GateBlocked, match='padded provenance'):
        audit_splits({'pool': [a]})


def test_missing_response_provenance_blocks():
    a = row(0)
    del a['response_id']
    with pytest.raises(GateBlocked, match='response_id'):
        audit_splits({'pool': [a]})


@pytest.mark.parametrize('function', ['temperature', 'threshold'])
def test_locked_labels_cannot_be_fitted_even_with_false_declaration(tmp_path, function):
    root, h = locked(tmp_path)
    labels = labels_for_role(root, h, 'locked_test')
    logits = np.ones((len(labels), 3))
    with pytest.raises(ValueError, match='locked test'):
        if function == 'temperature':
            fit_temperature(logits, labels, data_role='development')
        else:
            best_threshold(logits, labels, 1., 'contradiction', data_role='development')


def test_role_is_preserved_when_slicing_locked_labels():
    values = RoleLabels([0, 1, 2], 'locked_test')
    with pytest.raises(ValueError, match='locked test'):
        fit_temperature(np.ones((2, 3)), values[:2])


def test_unmarked_numerical_inputs_fail_closed():
    with pytest.raises(ValueError, match='explicit development'):
        fit_temperature(np.ones((3, 3)), np.array([0, 1, 2]))
    with pytest.raises(ValueError, match='explicit development'):
        best_threshold(np.ones((3, 3)), np.array([0, 1, 2]), 1., 'contradiction')


def test_development_role_is_the_only_accepted_tuning_role(tmp_path):
    root, h = locked(tmp_path)
    labels = labels_for_role(root, h, 'development')
    logits = np.eye(3)[labels] * 3
    threshold, _ = best_threshold(logits, labels, 1., 'contradiction', grid=np.array([.4, .5]))
    assert threshold in (.4, .5)


def test_standard_training_rejects_renamed_locked_rows_before_model_load(tmp_path):
    a, b = row(0), row(1)
    b['data_role'] = 'locked_test'
    for name, r in [('train.jsonl', a), ('dev.jsonl', b)]:
        (tmp_path / name).write_bytes(canonical(r) + b'\n')
    with pytest.raises(ValueError, match='locked test'):
        train(tmp_path, tmp_path / 'unused-output')
    assert not (tmp_path / 'unused-output').exists()


def test_standard_training_rejects_managed_dev_file_replacement(tmp_path):
    root, h = locked(tmp_path)
    (root / 'dev.jsonl').write_bytes((root / 'locked_test.jsonl').read_bytes())
    with pytest.raises(GateBlocked, match='mutation'):
        train(root, tmp_path / 'unused-output')


def test_bootstrap_is_reproducible_and_undefined_denominators_are_counted():
    logits = np.array([[4., 0., 0.], [3., 1., 0.], [0., 4., 0.], [0., 0., 4.]])
    labels, groups = np.array([0, 1, 1, 2]), ['a', 'a', 'b', 'c']
    args = dict(temperature=1., contradiction_threshold=.46, risk_threshold=.625, seed=19, replicates=60)
    assert bootstrap_intervals(logits, labels, groups, **args) == bootstrap_intervals(logits, labels, groups, **args)
    result = bootstrap_intervals(logits, labels, groups, **args)
    assert result['source_clusters'] == 3
    fa = result['intervals']['conditional_false_accept_rate']
    assert fa['valid_replicates'] + fa['undefined_replicates'] == 60


def test_operational_false_accepts_are_separate_from_argmax_quality():
    logits = np.log(np.array([[.7, .2, .1], [.7, .2, .1], [.1, .8, .1], [.1, .1, .8]]))
    result = metric_report(logits, np.array([0, 1, 1, 2]), temperature=1., contradiction_threshold=.46, risk_threshold=.625)
    assert result['model_quality']['confusion_matrix']['matrix'] == [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    p = result['operational_policy']
    assert p['hypothetical_bypass_count'] == 2
    assert p['conditional_false_accept_rate'] == .5
    assert p['coverage'] == .5
    assert p['contradiction_binary']['fn'] == 1
    assert p['fast_path_enabled'] is False


def test_no_bypass_and_no_contradictions_are_undefined_not_fabricated():
    result = metric_report(np.zeros((3, 3)), np.zeros(3, dtype=int), temperature=1., contradiction_threshold=1., risk_threshold=0.)
    assert result['operational_policy']['conditional_false_accept_rate'] is None
    assert result['operational_policy']['contradiction_binary']['recall'] is None


def test_missing_artifact_cli_is_blocked_not_fake_metrics(tmp_path, capsys):
    missing = str(tmp_path / 'missing.jsonl')
    assert main(['audit', '--train', missing, '--dev', missing, '--test', missing]) == 2
    result = json.loads(capsys.readouterr().err)
    assert result['status'] == 'INDEPENDENT_EVALUATION_BLOCKED'
    assert 'metrics' not in result


def test_no_exposure_record_blocks_before_predictor(tmp_path):
    root, h = locked(tmp_path)
    model = checkpoint(tmp_path)
    with pytest.raises(GateBlocked, match='exposure record missing'):
        evaluate_locked(root, h, model, tmp_path / 'evaluation', predictor=lambda r: pytest.fail('must not run'))
    assert not (tmp_path / 'evaluation').exists()


def test_historical_training_membership_overrides_new_clean_splits(tmp_path):
    model = checkpoint(tmp_path)
    history = {k: v for k, v in splits().items() if k != 'locked_test'}
    history['training'].append(deepcopy(splits()['locked_test'][0]))
    e = exposure(tmp_path, model, history)
    root, h = locked(tmp_path, exposure=e)
    bundle, rows = verify_bundle(root, h)
    with pytest.raises(GateBlocked, match='exposure overlaps'):
        verify_checkpoint_exposure(bundle, rows['locked_test'], file_hash(model / 'model.safetensors'))


def test_historical_train_dev_contamination_also_blocks(tmp_path):
    model = checkpoint(tmp_path)
    history = {k: v for k, v in splits().items() if k != 'locked_test'}
    history['training'].append(deepcopy(history['development'][0]))
    e = exposure(tmp_path, model, history)
    root, h = locked(tmp_path, exposure=e)
    bundle, rows = verify_bundle(root, h)
    with pytest.raises(GateBlocked, match='independent calibration'):
        verify_checkpoint_exposure(bundle, rows['locked_test'], file_hash(model / 'model.safetensors'))


def test_calibration_artifact_needs_its_own_history_link(tmp_path):
    model = checkpoint(tmp_path)
    e = exposure(tmp_path, model)
    data = json.loads(e.read_text())
    data['calibration_sha256'] = 'wrong'
    e.write_bytes(canonical(data))
    root, h = locked(tmp_path, exposure=e)
    with pytest.raises(GateBlocked, match='calibration not linked'):
        evaluate_locked(root, h, model, tmp_path / 'evaluation', predictor=lambda r: None)


def test_mutated_historical_exposure_is_rejected(tmp_path):
    model = checkpoint(tmp_path)
    e = exposure(tmp_path, model)
    root, h = locked(tmp_path, exposure=e)
    path = tmp_path / 'historical-training.jsonl'
    path.write_bytes(path.read_bytes() + b' ')
    bundle, rows = verify_bundle(root, h)
    with pytest.raises(GateBlocked, match='exposure artifact mutation'):
        verify_checkpoint_exposure(bundle, rows['locked_test'], file_hash(model / 'model.safetensors'))


def test_only_synthetic_fixture_may_use_injected_predictor(tmp_path):
    model = checkpoint(tmp_path)
    e = exposure(tmp_path, model)
    root, h = locked(tmp_path, exposure=e, kind='real')
    with pytest.raises(GateBlocked, match='injected predictors'):
        evaluate_locked(root, h, model, tmp_path / 'evaluation', predictor=lambda r: None)


def test_end_to_end_fixture_never_claims_real_independent_success(tmp_path):
    model = checkpoint(tmp_path)
    e = exposure(tmp_path, model)
    root, h = locked(tmp_path, exposure=e)
    def predict(rows):
        assert (tmp_path / 'evaluation/evaluation-plan.json').is_file()
        labels = np.array([r['label_id'] for r in rows])
        return np.eye(3)[labels] * 3, labels
    result = evaluate_locked(root, h, model, tmp_path / 'evaluation', predictor=predict, bootstrap_replicates=30)
    assert result['status'] == 'SYNTHETIC_FRAMEWORK_TEST_ONLY'
    assert result['production_ready'] is False
    assert result['live_verifier_n8n'] == 'unverified'
    assert (tmp_path / 'evaluation/evaluation-plan.json').exists()
    assert len((tmp_path / 'evaluation/predictions.jsonl').read_text().splitlines()) == 6


def test_mid_evaluation_checkpoint_mutation_never_publishes_success(tmp_path):
    model = checkpoint(tmp_path)
    e = exposure(tmp_path, model)
    root, h = locked(tmp_path, exposure=e)
    def predict(rows):
        (model / 'model.safetensors').write_text('mutated-placeholder')
        labels = np.array([r['label_id'] for r in rows])
        return np.eye(3)[labels] * 3, labels
    with pytest.raises(GateBlocked, match='changed during inference'):
        evaluate_locked(root, h, model, tmp_path / 'evaluation', predictor=predict, bootstrap_replicates=4)
    assert not (tmp_path / 'evaluation/evaluation.json').exists()
