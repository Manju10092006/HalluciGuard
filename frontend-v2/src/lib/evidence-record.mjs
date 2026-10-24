const text = value => typeof value === 'string' ? value.trim() : '';

// Display only provenance present in the evidence record. Retrieval does not
// establish that a source is authoritative or that its passage entails a claim.
export function evidenceRecordView(source = {}) {
  const rawDirection = text(source.entailment_label).toLowerCase().split('.').at(-1);
  const direction = rawDirection || ({ supports: 'entailment', contradicts: 'contradiction', context: 'neutral' }[source.relationship] || '');
  const relation = direction === 'entailment'
    ? { label: 'Supports claim', tone: 'supported', status: 'Entailment' }
    : direction === 'contradiction'
      ? { label: 'Contradicts claim', tone: 'contradicted', status: 'Contradiction' }
      : { label: 'Context only', tone: 'insufficient', status: direction === 'neutral' ? 'Neutral / context only' : 'Not provided' };
  let externalUrl = '';
  try {
    const url = new URL(source.url);
    if (url.protocol === 'https:' || url.protocol === 'http:') externalUrl = url.href;
  } catch {}
  return {
    ...relation,
    attribution: text(source.source) || text(source.source_id) || 'Not provided',
    retrievalMethod: text(source.retrieval_method),
    externalUrl,
  };
}
