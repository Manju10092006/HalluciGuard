import test from 'node:test';
import assert from 'node:assert/strict';
import { evidenceRecordView } from './src/lib/evidence-record.mjs';

test('neutral evidence cannot be presented as factual entailment', () => {
  const record = evidenceRecordView({ entailment_label: 'EntailmentLabel.neutral', relationship: 'supports', source: 'Wikipedia' });
  assert.equal(record.label, 'Context only');
  assert.equal(record.status, 'Neutral / context only');
  assert.equal(record.tone, 'insufficient');
});

test('missing provenance does not invent authority or a retrieval method', () => {
  const record = evidenceRecordView({ relationship: 'context' });
  assert.equal(record.attribution, 'Not provided');
  assert.equal(record.retrievalMethod, '');
  assert.equal(record.label, 'Context only');
});

test('API relationship values retain their distinct evidence directions', () => {
  assert.equal(evidenceRecordView({ relationship: 'supports' }).status, 'Entailment');
  assert.equal(evidenceRecordView({ relationship: 'contradicts' }).status, 'Contradiction');
  assert.equal(evidenceRecordView({ relationship: 'context' }).status, 'Neutral / context only');
});

test('actual attribution and contradiction direction survive display mapping', () => {
  const record = evidenceRecordView({ relationship: 'contradicts', source_id: 'wiki:123', retrieval_method: 'tavily_web', url: 'https://example.org/source' });
  assert.equal(record.status, 'Contradiction');
  assert.equal(record.attribution, 'wiki:123');
  assert.equal(record.retrievalMethod, 'tavily_web');
  assert.equal(record.externalUrl, 'https://example.org/source');
});

test('external evidence link cannot execute an untrusted URL scheme', () => {
  for (const url of ['javascript:alert(1)', 'data:text/html,test', 'file:///private']) {
    assert.equal(evidenceRecordView({ url }).externalUrl, '');
  }
});
