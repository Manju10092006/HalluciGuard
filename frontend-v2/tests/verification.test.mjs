import { test } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeVerification, safeSourceUrl, verifyMessage } from '../src/lib/verification.js';

const response = (overrides = {}) => ({ execution_id: 'test', terminal_status: 'accepted', verification_status: 'verified_and_accepted', final_response: 'Verified answer.', verifier: { claim_evidence: [{ claim_id: 'c1', claim_text: 'A claim.', verdict: 'verified', evidence: [] }] }, ...overrides });

test('a supported badge requires an accepted answer with every claim verified', () => {
  assert.equal(normalizeVerification(response()).status, 'SUPPORTED');
  for (const verdict of ['unverified', 'conflicted', 'contradicted']) {
    const result = response({ verifier: { claim_evidence: [{ verdict }] } });
    assert.notEqual(normalizeVerification(result).status, 'SUPPORTED');
  }
  assert.notEqual(normalizeVerification(response({ verifier: { claim_evidence: [] } })).status, 'SUPPORTED');
  assert.notEqual(normalizeVerification(response({ terminal_status: 'human_review' })).status, 'SUPPORTED');
});

test('unreleased drafts and unverified corrections never appear as verified answers', () => {
  const result = response({ final_response: '', draft_response: 'Unsafe draft', correction_result: { corrected_text: 'Unverified repair' }, reverification_result: { passed: false } });
  const message = normalizeVerification(result);
  assert.equal(message.correction, null);
  assert.ok(!message.text.includes('Unsafe draft'));
});

test('source relationships preserve neutral context and reject executable links', () => {
  const message = normalizeVerification(response({ verifier: { claim_evidence: [{ verdict: 'unverified', evidence: [{ url: 'javascript:alert(1)', classification: 'neutral', snippet: 'A passage.' }] }] } }));
  assert.equal(message.sources[0].url, null);
  assert.equal(message.sources[0].relationshipTone, 'context');
  assert.equal(safeSourceUrl('https://example.com/source'), 'https://example.com/source');
});

test('backend failures reject without fabricated verdicts or sources', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({ detail: 'Backend unavailable' }), { status: 503 });
  try { await assert.rejects(verifyMessage('Java was created by Snehith.'), /Backend unavailable/); }
  finally { globalThis.fetch = original; }
});

test('real transport sends the backend JWT and bounded conversation history', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    assert.equal(url, '/api/backend/verify');
    assert.equal(options.headers.get('authorization'), 'Bearer server-jwt');
    const body = JSON.parse(options.body);
    assert.equal(body.user_query, 'A question');
    assert.equal(body.conversation_history.length, 1);
    assert.equal(body.conversation_history[0].role, 'user');
    return new Response(JSON.stringify(response()), { status: 200 });
  };
  try { await verifyMessage('A question', { token: 'server-jwt', history: [{ sender: 'user', text: 'Earlier question' }, { sender: 'assistant', text: 'Unverified draft', status: 'UNCERTAIN' }] }); }
  finally { globalThis.fetch = original; }
});
