import { backendRequest } from './backend.js';

export function safeSourceUrl(value) {
  try {
    const url = new URL(value);
    return ['https:', 'http:'].includes(url.protocol) ? url.href : null;
  } catch { return null; }
}

export function normalizeVerification(result) {
  if (!result || !result.terminal_status || !result.verification_status) {
    throw new Error('The backend returned an incomplete verification result.');
  }
  const reports = result.verifier?.claim_evidence || [];
  const accepted = result.terminal_status === 'accepted' && result.verification_status === 'verified_and_accepted' && reports.length > 0 && reports.every(c => c.verdict === 'verified');
  const contradicted = reports.some(c => c.verdict === 'contradicted');
  const conflicted = reports.some(c => c.verdict === 'conflicted');
  const status = accepted ? 'SUPPORTED' : conflicted ? 'UNCERTAIN' : contradicted ? 'CONTRADICTED' : 'INSUFFICIENT_EVIDENCE';
  const sources = reports.flatMap((claim, claimIndex) => (claim.evidence || []).map((ev, index) => {
    const url = safeSourceUrl(ev.url);
    // Use the backend's evidence classification, retaining neutral as context.
    const relation = ev.classification || ev.entailment_label;
    const tone = ['supporting', 'entailment'].includes(relation) ? 'supported' : ['contradicting', 'contradiction'].includes(relation) ? 'contradicted' : 'context';
    return {
      id: `${claim.claim_id || claimIndex}-${index}`, title: ev.title || ev.source || 'Retrieved source',
      domain: url ? new URL(url).hostname : ev.source || 'Source unavailable', url,
      excerpt: ev.snippet || '', relationshipTone: tone,
      relationship: tone === 'supported' ? 'Supports claim' : tone === 'contradicted' ? 'Contradicts claim' : 'Context only',
      extractionMethod: 'Retrieved passage', sourceType: ev.source || 'Retrieved source',
    };
  }));
  const correction = result.correction_result;
  const repaired = accepted && result.reverification_result?.passed === true && correction?.corrected_text;
  return {
    id: result.execution_id || globalThis.crypto.randomUUID(), sender: 'assistant',
    text: result.final_response || 'No answer was released. Please retry verification.',
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    status, statusTitle: repaired ? 'CORRECTED & VERIFIED' : status.replaceAll('_', ' '),
    statusDescription: accepted ? 'The backend Judge accepted this answer after evidence verification.' : 'The answer was not released as verified. Review the evidence and pipeline details.',
    sources,
    evidenceSummary: { totalSources: sources.length, supportingCount: sources.filter(s => s.relationshipTone === 'supported').length, contradictingCount: sources.filter(s => s.relationshipTone === 'contradicted').length },
    claims: reports.map((c, i) => ({ number: String(i + 1).padStart(2, '0'), text: c.claim_text, status: c.verdict, tone: c.verdict === 'verified' ? 'supported' : c.verdict === 'contradicted' ? 'contradicted' : 'context', verdict: c.explanation, evidenceExcerpt: c.evidence?.[0]?.snippet })),
    correction: repaired ? { original: result.draft_response, corrected: correction.corrected_text, verificationNote: 'Accepted after independent re-verification.' } : null,
    pipeline: (result.trace || []).map((event, index) => ({ id: `${event.node || event.agent || 'stage'}-${index}`, name: event.node || event.agent || 'Pipeline stage', detail: event.status || event.detail || 'Recorded', latency_ms: event.latency_ms })),
    detector: result.detector,
    executionId: result.execution_id,
  };
}

export async function verifyMessage(text, { token, history = [] } = {}) {
  const result = await backendRequest('/verify', {
    method: 'POST', token,
    body: JSON.stringify({
      user_query: text, domain: 'general', generation_mode: 'normal',
      conversation_history: history.filter(m => m.sender === 'user' || m.status === 'SUPPORTED').slice(-30).map(m => ({ role: m.sender === 'user' ? 'user' : 'assistant', content: m.text.slice(0, 20000) })),
    }),
  });
  return normalizeVerification(result);
}
