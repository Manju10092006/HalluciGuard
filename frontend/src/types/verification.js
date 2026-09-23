/**
 * Verification status constants and data models for HalluciGuard
 */

export const VERIFICATION_STATUS = {
  SUPPORTED: 'SUPPORTED',
  CONTRADICTED: 'CONTRADICTED',
  INSUFFICIENT_EVIDENCE: 'INSUFFICIENT_EVIDENCE',
  UNCERTAIN: 'UNCERTAIN',
  NEEDS_CORRECTION: 'NEEDS_CORRECTION',
  VERIFIED: 'VERIFIED',
}

export const VERIFICATION_STATES = {
  EMPTY: 'EMPTY',
  TYPING: 'TYPING',
  SUBMITTING: 'SUBMITTING',
  ANALYZING: 'ANALYZING',
  RETRIEVING_EVIDENCE: 'RETRIEVING_EVIDENCE',
  VERIFYING: 'VERIFYING',
  CORRECTION_REQUIRED: 'CORRECTION_REQUIRED',
  REVERIFICATION: 'REVERIFICATION',
  VERIFIED: 'VERIFIED',
  INSUFFICIENT_EVIDENCE: 'INSUFFICIENT_EVIDENCE',
  ERROR: 'ERROR',
}

export const AGENT_PIPELINE = [
  { id: 'detector', name: 'Detector', role: 'Deconstructs response into atomic claims with full context preservation' },
  { id: 'verifier', name: 'Verifier', role: 'Retrieves primary sources and evaluates claim entailment against passages' },
  { id: 'judge', name: 'Judge', role: 'Synthesizes NLI verdicts, weights source authority and flags contradictions' },
  { id: 'corrector', name: 'Corrector', role: 'Rewrites contradicted statements using grounded evidence' },
  { id: 'reverifier', name: 'ReVerifier', role: 'Runs secondary consistency checks on the corrected answer' },
  { id: 'memory', name: 'Memory', role: 'Records verified claims and audit trail in persistent provenance index' },
]
