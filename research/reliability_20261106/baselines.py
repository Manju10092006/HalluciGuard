"""Bounded common interfaces. Named literature adapters are approximations.

The callbacks use the existing project generator, verifier and graph. This
module does not implement a second HalluciGuard system. It never treats an
unavailable judgment or provider failure as a successful release.
"""
from dataclasses import dataclass, field
import json
import re
import time
from typing import Awaitable, Callable


@dataclass
class Case:
    id: str
    question: str
    draft: str
    evidence: list[dict] = field(default_factory=list)


@dataclass
class Budget:
    max_calls: int = 8
    max_output_tokens: int = 512
    max_prompt_chars: int = 24000
    calls: int = 0


class Backend:
    def __init__(self, generate, *, verify=None, nli=None, graph=None, budget=None):
        self.generate = generate
        self.verify = verify
        self.nli = nli
        self.graph = graph
        self.budget = budget or Budget()
        self.trace = []

    async def call(self, prompt, temperature=0.0):
        if self.budget.calls >= self.budget.max_calls:
            raise RuntimeError('CALL_BUDGET_EXHAUSTED')
        if len(prompt) > self.budget.max_prompt_chars:
            raise RuntimeError('PROMPT_BUDGET_EXCEEDED')
        self.budget.calls += 1
        start = time.perf_counter()
        row = {'prompt': prompt, 'temperature': temperature,
               'max_tokens': self.budget.max_output_tokens}
        self.trace.append(row)
        try:
            result = await self.generate(prompt, temperature, self.budget.max_output_tokens)
            row['result'] = result
            if result.get('status') != 'success' or not result.get('text', '').strip():
                raise RuntimeError('GENERATION_FAILED')
            return result['text'].strip()
        finally:
            row['seconds'] = time.perf_counter() - start


def questions(text, maximum=3):
    """Reject malformed planning output; never silently invent questions."""
    clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    values = json.loads(clean)
    if not isinstance(values, list) or not values or not all(isinstance(v, str) and v.strip() for v in values):
        raise ValueError('INVALID_QUESTION_PLAN')
    return list(dict.fromkeys(v.strip() for v in values))[:maximum]


async def run(method: str, case: Case, backend: Backend):
    start = time.perf_counter()
    result = {'method': method, 'id': case.id, 'status': 'running', 'answer': '',
              'human_factuality': None}
    try:
        q, draft = case.question, case.draft
        evidence = json.dumps(case.evidence, ensure_ascii=True)
        if method == 'plain_shared_draft':
            answer = draft
        elif method == 'self_critique':
            answer = await backend.call(f'Question: {q}\nDraft: {draft}\nCheck the factual assertions using your knowledge. Correct errors, preserve correct content, and return only the final answer. Say when you do not know.')
        elif method in {'basic_rag', 'rag_claim_verification'}:
            answer = await backend.call(f'Question: {q}\nEvidence: {evidence}\nAnswer only from these passages. If they are insufficient, say so. Return only the answer.')
            if method == 'rag_claim_verification':
                if backend.verify is None:
                    raise RuntimeError('VERIFIER_CALLBACK_REQUIRED')
                reports = await backend.verify(q, answer)
                result['reports'] = reports
                if not reports or any(r.get('verdict') != 'verified' for r in reports):
                    result['status'] = 'withheld'
        elif method == 'cove_style_factored':
            plan = await backend.call(f'Question: {q}\nDraft: {draft}\nPlan up to three independent factual questions that can check the draft. Return ONLY a JSON array of question strings. Do not answer them.')
            checks = []
            for check in questions(plan):
                # Factored verification deliberately receives neither draft nor
                # previous check answers. It still shares the same model.
                reply = await backend.call(f'Answer this factual question concisely from your knowledge. State uncertainty when necessary.\nQuestion: {check}')
                checks.append({'question': check, 'answer': reply})
            result['verification_answers'] = checks
            answer = await backend.call(f'Question: {q}\nDraft: {draft}\nIndependent checks: {json.dumps(checks)}\nUse these checks to correct the draft while preserving correct content. Return only the final answer, including uncertainty when needed.')
        elif method == 'rarr_style_frozen_corpus':
            plan = await backend.call(f'Draft: {draft}\nCreate up to three factual verification questions covering this draft. Return ONLY a JSON array of strings.')
            checks = []
            answer = draft
            for check in questions(plan):
                # Fixed source pool controls retrieval access across conditions.
                # This is NOT the original RARR web search or trained editor.
                tokens = set(re.findall(r'\w+', check.casefold()))
                ranked = sorted(case.evidence, key=lambda p: len(tokens & set(re.findall(r'\w+', p.get('snippet', '').casefold()))), reverse=True)[:3]
                response = await backend.call(f'Check question: {check}\nCurrent answer: {answer}\nEvidence: {json.dumps(ranked)}\nReturn ONLY a JSON object with keys supported (boolean) and reason (string). Decide whether the relevant assertion is supported by these passages, not by outside knowledge.')
                obj = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', response.strip()))
                if not isinstance(obj.get('supported'), bool):
                    raise ValueError('INVALID_AGREEMENT_RESPONSE')
                checks.append({'question': check, 'evidence': ranked, 'assessment': obj})
                if not obj['supported']:
                    answer = await backend.call(f'Question: {q}\nCurrent answer: {answer}\nIssue: {obj.get("reason", "")}\nEvidence: {json.dumps(ranked)}\nMake the smallest factual edit justified by the evidence. Preserve unaffected content and style. If evidence is insufficient, express uncertainty. Return only the revised answer.')
            result['research_and_revision'] = checks
        elif method == 'selfcheck_style_nli':
            if backend.nli is None:
                raise RuntimeError('NLI_CALLBACK_REQUIRED')
            samples = [await backend.call(f'Answer the question concisely.\nQuestion: {q}', temperature=1.0) for _ in range(3)]
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', draft) if s.strip()]
            scores = []
            for sentence in sentences:
                pairs = await backend.nli(sentence, samples)
                if len(pairs) != len(samples) or any(x.get('degraded') for x in pairs):
                    raise RuntimeError('INVALID_SELF_CHECK_SCORES')
                scores.append({'sentence': sentence, 'pairs': pairs,
                               'contradiction_mean': sum(x['contradiction_score'] for x in pairs)/len(pairs)})
            result.update(status='scored_only', samples=samples, sentence_scores=scores)
            # The published family is a detector. Do not invent a threshold or
            # turn its uncertainty into an unvalidated answer release policy.
            answer = draft
        elif method in {'halluciguard_original', 'halluciguard_candidate'}:
            if backend.graph is None:
                raise RuntimeError('VERSIONED_GRAPH_CALLBACK_REQUIRED')
            state = await backend.graph(case, method)
            result['graph_result'] = state
            result['status'] = state['status']
            answer = state.get('answer', '')
        else:
            raise ValueError('UNKNOWN_BASELINE')
        result['answer'] = answer
        if result['status'] == 'running':
            result['status'] = 'produced'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = type(exc).__name__ + ': ' + str(exc)
    result.update(seconds=time.perf_counter()-start, calls=backend.budget.calls,
                  trace=backend.trace)
    return result
