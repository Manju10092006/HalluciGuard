"""Opt-in arbitration candidates. The historical structured policy is unchanged.

These policies express evidence handling hypotheses, not validated improvements.
"""
import re
from .relation_verifier import RelationVerifier, RelationCheckResult, Triple


class NoScopedRelationVerifier(RelationVerifier):
    def verify_relation(self, claim_text, evidence_passages):
        return RelationCheckResult(mismatch_detail='Scoped arbitration disabled')

    def is_temporally_inapplicable(self, claim_text, passage):
        return False


class NliTemporalRelationVerifier(NoScopedRelationVerifier):
    is_temporally_inapplicable = RelationVerifier.is_temporally_inapplicable


class SoftRelationVerifier(RelationVerifier):
    """Retain decisive comparisons, defer inconclusive extraction to NLI.

Unknown structure never raises scores. A limited location construction addresses
    the archived neutral-NLI failure. Multi-valued alternatives alone cannot
    prove contradiction. Negated, hedged and attributed statements defer to NLI.
    """
    allow_unresolved_nli = True
    _LOCATION = re.compile(
        r"^([A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*){0,4})\s+"
        r"(lies|lie|does not lie|do not lie)\s+in\s+"
        r"([A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*){0,4})$"
    )

    def extract_triples(self, text):
        triples = super().extract_triples(text)
        for sentence in re.split(r'[.!?\n]+', str(text or '')):
            sentence = sentence.strip()
            match = self._LOCATION.fullmatch(sentence)
            if match:
                triples.append(Triple(subject=self._normalize_name(match[1]), relation='location_of',
                                      object=self._normalize_name(match[3]), negated='not' in match[2],
                                      qualifiers=['literal_location'], raw_text=sentence))
        return triples

    def verify_relation(self, claim_text, evidence_passages):
        result = super().verify_relation(claim_text, evidence_passages)
        if (result.status in {'OBJECT_MISMATCH','RELATION_MISMATCH'}
                and result.claim_triple and result.claim_triple.relation in {'created_by','leads','location_of'}):
            # Another cofounder or containing region need not refute this value.
            return result.model_copy(update={'status':'NO_TRIPLE_EXTRACTED',
                'combination_rule_applied':'', 'mismatch_detail':'Alternative multi-value relation defers to NLI'})
        return result


def make_relation_verifier(policy):
    variants = {'structured':RelationVerifier, 'no_scoped':NoScopedRelationVerifier,
                'nli_temporal':NliTemporalRelationVerifier, 'soft':SoftRelationVerifier}
    if policy not in variants:
        raise ValueError(f'Unknown relation policy: {policy}')
    return variants[policy]()
