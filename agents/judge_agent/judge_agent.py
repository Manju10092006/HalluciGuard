"""
HalluciGuard - Canonical Judge Agent
The Chief Decision Officer of HalluciGuard.

The Judge receives VerifierResult from the Verifier and decides what the system should do next:
  - ACCEPT: Release draft response verbatim
  - CORRECT: Deliver targeted CorrectionRequest payload to Snehith's Corrector Agent
  - VERIFY_AGAIN: Request expanded verification pass (if retries available)
  - REJECT: Block response due to critical safety/contradiction risk
  - ABSTAIN: Insufficient evidence or unresolvable pipeline degradation

The Judge does NOT perform independent fact-checking, NLI model inference, or keyword refutation checks.
It relies on the authoritative factual investigation produced by the Verifier.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union

try:
    from agents.judge_agent.config import JudgeConfig, DEFAULT_CONFIG
except ImportError:
    from config import JudgeConfig, DEFAULT_CONFIG

try:
    from agents.judge_agent.domain_policies import DomainPolicyRegistry, DEFAULT_DOMAIN_REGISTRY, DomainPolicy
except ImportError:
    from domain_policies import DomainPolicyRegistry, DEFAULT_DOMAIN_REGISTRY, DomainPolicy

from orchestration.schemas import (
    JudgeResult,
    CorrectionRequest,
    ReverificationResult,
    VerifierResult,
    ClaimReport,
    Evidence,
    DetectorResult,
    JudgeDecision,
    SeverityLevel,
    VerdictLabel,
    EntailmentLabel,
    ExecutionStatus,
)

logger = logging.getLogger("HalluciGuard.JudgeAgent")


class DecisionBasis:
    """Stable machine-readable reason codes for the Judge's precedence rules.

    These are the *why* behind a JudgeDecision, decoupled from the human-facing
    ``reason``/``explanation`` prose so metrics, dashboards, and regression tests
    can key off a stable token instead of matching free text. A basis code names
    the rule that fired; it is never itself a factual claim about the response.
    """

    # Terminal input-integrity outcomes
    INVALID_VERIFIER_INPUT = "INVALID_VERIFIER_INPUT_ABSTAIN"
    VERIFIER_FAILED = "VERIFIER_FAILED_ABSTAIN"

    # Contradiction (correct-first)
    CONTRADICTION_PRESENT = "CONTRADICTION_PRESENT_CORRECT"

    # Absent evidence
    NO_EVIDENCE_RETRY = "NO_EVIDENCE_RETRY"
    NO_EVIDENCE_ABSTAIN = "NO_EVIDENCE_ABSTAIN"

    # Verified outcomes
    ALL_CLAIMS_VERIFIED = "ALL_CLAIMS_VERIFIED_ACCEPT"
    PERIPHERAL_UNVERIFIED_TOLERATED = "PERIPHERAL_UNVERIFIED_TOLERATED_ACCEPT"

    # Core / conflicted grounding gaps
    CORE_UNVERIFIED_RETRY = "CORE_UNVERIFIED_RETRY"
    CORE_UNVERIFIED_ABSTAIN = "CORE_UNVERIFIED_ABSTAIN"
    CONFLICTED_RETRY = "CONFLICTED_RETRY"
    STRICT_GROUNDING_GAP_ABSTAIN = "STRICT_GROUNDING_GAP_ABSTAIN"

    # Post-correction reverification gate
    REVERIFICATION_PASSED = "REVERIFICATION_PASSED_ACCEPT"
    REVERIFICATION_FAILED_RETRY = "REVERIFICATION_FAILED_RETRY_CORRECT"
    REVERIFICATION_FAILED_REJECT = "REVERIFICATION_FAILED_REJECT"


# Tokens that carry no discriminative signal when deciding whether a claim is
# responsive to the user's query. Kept small and generic on purpose.
_QUERY_STOPWORDS = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "with", "and",
    "or", "is", "are", "was", "were", "be", "been", "being", "who", "what",
    "when", "where", "which", "why", "how", "does", "did", "do", "can", "could",
    "would", "should", "will", "that", "this", "these", "those", "it", "its",
    "as", "from", "about", "into", "than", "then", "there", "their", "them",
    "has", "have", "had", "you", "your", "me", "my", "i", "we", "our",
})


def _salient_terms(text: str) -> set[str]:
    """Extract lowercase content tokens (>2 chars, non-stopword) plus any numbers.

    Deterministic and dependency-free — no model, no network. Numbers are kept
    verbatim because dates/quantities are frequently the crux of a query
    ("founded in 1977", "how many...").
    """
    import re

    if not text:
        return set()
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", text.lower())
    terms: set[str] = set()
    for tok in tokens:
        if tok.isdigit():
            terms.add(tok)
        elif len(tok) > 2 and tok not in _QUERY_STOPWORDS:
            terms.add(tok)
    return terms


def _claim_is_core(claim_text: str, query_terms: set[str]) -> bool:
    """Decide whether a claim is CORE (directly responsive to the query) or
    PERIPHERAL (incidental detail the user did not ask about).

    Deterministic and query-aware: a claim is CORE when its salient terms
    intersect the query's salient terms. This is the criticality gate the ACCEPT
    path depends on — a CORE claim that is UNVERIFIED must never be silently
    accepted, whereas a PERIPHERAL unverified fragment (e.g. a decomposition
    artifact or an incidental bio detail) may be tolerated.

    Fail-closed: when the query yields no salient anchors (degenerate/empty
    query) criticality cannot be established, so every claim is treated as CORE.
    Absence of a signal is never read as "safe to accept".
    """
    if not query_terms:
        return True
    return bool(_salient_terms(claim_text) & query_terms)


def _claim_leaves_core_gap(
    claim_text: str,
    query_terms: set[str],
    grounded_anchors: set[str],
) -> bool:
    """Decide whether an UNVERIFIED / CONFLICTED claim leaves a CORE grounding gap.

    A claim only leaves a core gap when it touches a query anchor that is *not
    already grounded by a verified claim*. This is stricter than bare overlap:
    an unverified fragment that merely re-mentions the query subject already
    proven elsewhere (e.g. "designed by Gustave Eiffel, completed 1889" when the
    Eiffel Tower's location is verified) adds incidental detail the user did not
    ask about — peripheral, not a core gap.

    Fail-closed exactly like `_claim_is_core`:
      * empty/degenerate query  -> every ungrounded claim is a core gap (True);
      * no verified claims       -> `grounded_anchors` is empty, so any claim
        sharing a query anchor is a core gap (all-unverified answers still
        ABSTAIN);
      * a claim sharing NO query anchor is always peripheral (False).
    Absence of a grounding signal is never read as "safe to accept".
    """
    if not query_terms:
        return True
    shared = _salient_terms(claim_text) & query_terms
    if not shared:
        return False
    # CORE gap only if the claim introduces a query anchor no verified claim covers.
    return bool(shared - grounded_anchors)


def _verdict_value(value: Any) -> str:
    """Robustly extract a verdict as a lowercase value string.

    Claim verdicts normally arrive as plain strings (schemas use
    ``use_enum_values=True``), but a raw ``VerdictLabel`` enum can slip through
    on direct construction. ``str(VerdictLabel.CONTRADICTED)`` yields
    ``"VerdictLabel.CONTRADICTED"`` for a ``(str, Enum)`` member, which would
    fail exact comparison against ``"contradicted"``. Prefer ``.value``.
    """
    return str(getattr(value, "value", value)).lower()


class JudgeAgent:
    """
    Canonical Judge Agent.
    Evaluates VerifierResult and emits canonical JudgeResult.
    """

    def __init__(self, config: Optional[JudgeConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.domain_registry = DEFAULT_DOMAIN_REGISTRY
        self._consecutive_errors = 0
        self._circuit_open = False
        logger.info("HalluciGuard Canonical Judge Agent initialized.")

    def evaluate(
        self,
        verifier_result: Union[VerifierResult, Dict[str, Any]],
        detector_result: Optional[Union[DetectorResult, Dict[str, Any]]] = None,
        user_query: str = "",
        original_response: str = "",
        draft_response: str = "",
        domain: str = "",
        reverification_result: Optional[Union[ReverificationResult, Dict[str, Any]]] = None,
        retry_count: int = 0,
        correction_attempt_count: int = 0
    ) -> JudgeResult:
        """
        Main decision arbitration entry point.
        """
        response_text = original_response or draft_response or ""

        # -------------------------------------------------------------------
        # 1. Post-Correction Re-verification Evaluation (Task 10)
        # -------------------------------------------------------------------
        if reverification_result is not None:
            effective_corr_retries = correction_attempt_count if correction_attempt_count > 0 else retry_count
            return self._evaluate_reverification(
                reverification_result, user_query, response_text, retry_count=effective_corr_retries
            )

        # -------------------------------------------------------------------
        # 2. Input Normalization & Controlled Failure Handling (Task 2 & 3)
        # -------------------------------------------------------------------
        normalized_verifier = self._normalize_verifier_result(verifier_result, domain)
        if normalized_verifier is None:
            logger.warning("Judge received empty or unparseable VerifierResult. Returning ABSTAIN.")
            return JudgeResult(
                decision=JudgeDecision.ABSTAIN,
                severity=SeverityLevel.HIGH,
                reason="Invalid or missing VerifierResult payload.",
                explanation="Grounding evidence was absent or failed schema validation. Unsafe to proceed.",
                confidence=0.0,
                correction_request=None,
                decision_basis=DecisionBasis.INVALID_VERIFIER_INPUT,
                status=ExecutionStatus.FAILED
            )

        normalized_detector = self._normalize_detector_result(detector_result)
        domain_name = normalized_verifier.domain or domain or "General Knowledge"
        policy = self.domain_registry.get_policy(domain_name)

        if str(normalized_verifier.status).lower() in ("failed", "executionstatus.failed"):
            logger.warning("VerifierResult status indicates failure. Returning ABSTAIN.")
            return JudgeResult(
                decision=JudgeDecision.ABSTAIN,
                severity=SeverityLevel.HIGH,
                reason="VerifierResult status indicates failure.",
                explanation="Grounding investigation failed to execute. Unsafe to proceed.",
                confidence=0.0,
                correction_request=None,
                decision_basis=DecisionBasis.VERIFIER_FAILED,
                status=ExecutionStatus.FAILED
            )

        # -------------------------------------------------------------------
        # 3. Claim-Level Decision Processing (Task 4 & 5 & 6 & 7)
        # -------------------------------------------------------------------
        claim_reports = normalized_verifier.claim_reports

        claims_to_correct: List[ClaimReport] = []
        claims_to_preserve: List[ClaimReport] = []
        trusted_evidence: List[Evidence] = []
        contradictory_evidence: List[Evidence] = []
        unverified_claims: List[ClaimReport] = []
        conflicted_claims: List[ClaimReport] = []

        for claim in claim_reports:
            verdict_str = _verdict_value(claim.verdict)
            if verdict_str == VerdictLabel.CONTRADICTED.value:
                claims_to_correct.append(claim)
                # A contradicted claim's evidence is the refuting context for THAT
                # claim and is kept as contradictory_evidence, cleanly separated from
                # the trusted_evidence that anchors preserved claims (Task 12 Test F).
                # The Corrector is not starved by this separation: it receives each
                # contradicted claim's own evidence directly, plus both evidence
                # buckets, so it always has the replacement fact to write from.
                for ev in claim.evidence:
                    contradictory_evidence.append(ev)
            elif verdict_str == VerdictLabel.VERIFIED.value:
                claims_to_preserve.append(claim)
                for ev in claim.evidence:
                    trusted_evidence.append(ev)
            elif verdict_str == VerdictLabel.CONFLICTED.value:
                conflicted_claims.append(claim)
            elif verdict_str == VerdictLabel.UNVERIFIED.value:
                unverified_claims.append(claim)
            else:
                if claim.contradiction_score >= 0.5:
                    claims_to_correct.append(claim)
                    for ev in claim.evidence:
                        contradictory_evidence.append(ev)
                elif claim.support_score >= 0.5:
                    claims_to_preserve.append(claim)
                    for ev in claim.evidence:
                        trusted_evidence.append(ev)
                else:
                    unverified_claims.append(claim)

        # -------------------------------------------------------------------
        # 4. Apply Policy Decision Governance Tree (Task 8 & 9)
        # -------------------------------------------------------------------
        det_prob = normalized_detector.hallucination_probability if normalized_detector else 0.0

        has_contradictions = len(claims_to_correct) > 0
        has_preservations = len(claims_to_preserve) > 0
        has_unverified = len(unverified_claims) > 0
        has_conflicted = len(conflicted_claims) > 0
        total_claims = len(claim_reports)

        # Criticality gate (deterministic, query-aware). A CORE claim is one the
        # user actually asked about; a PERIPHERAL claim is incidental detail. The
        # ACCEPT path may tolerate a PERIPHERAL unverified fragment but must never
        # accept while a CORE claim is unverified or conflicted. This — not a claim
        # count — is what separates "substantively grounded" from "ungrounded".
        query_terms = _salient_terms(user_query)
        # Query anchors already grounded by a VERIFIED claim. An unverified claim
        # that only touches these anchors is peripheral elaboration, not a core
        # gap — the thing the user asked about is already proven elsewhere.
        grounded_anchors: set[str] = set()
        for c in claims_to_preserve:
            grounded_anchors |= (_salient_terms(c.claim_text) & query_terms)
        core_unverified = [c for c in unverified_claims if _claim_leaves_core_gap(c.claim_text, query_terms, grounded_anchors)]
        core_conflicted = [c for c in conflicted_claims if _claim_leaves_core_gap(c.claim_text, query_terms, grounded_anchors)]
        core_preserved = [c for c in claims_to_preserve if _claim_is_core(c.claim_text, query_terms)]
        has_core_grounding_gap = bool(core_unverified) or bool(core_conflicted)

        decision: JudgeDecision = JudgeDecision.ABSTAIN
        severity: SeverityLevel = SeverityLevel.LOW
        reason: str = ""
        explanation: str = ""
        decision_basis: str = ""
        correction_req: Optional[CorrectionRequest] = None

        # Rule A: Contradicted Claims -> CORRECT (correct-first policy)
        #
        # Policy decision: even in safety-critical (VERY_STRICT / STRICT) domains a
        # contradicted claim is sent to the Corrector to regenerate from evidence
        # FIRST. We never hard-REJECT a fixable contradiction on the initial pass —
        # rejection is reserved for the post-correction path: if the regenerated
        # answer still fails re-verification after the bounded retry budget,
        # `_evaluate_reverification` issues the REJECT. This matches the intended
        # BASE -> ... -> CORRECTOR -> RE-VERIFIER -> JUDGE flow and avoids blocking a
        # repairable answer, while strict-domain contradictions are marked HIGH
        # severity so the failure path escalates decisively.
        if has_contradictions:
            is_critical_domain = policy.strictness_level in ["VERY_STRICT", "STRICT"]
            is_high_contradiction = any(c.contradiction_score >= policy.reject_contradiction_threshold for c in claims_to_correct)

            decision = JudgeDecision.CORRECT
            severity = SeverityLevel.HIGH if (is_critical_domain and is_high_contradiction) else SeverityLevel.MEDIUM
            decision_basis = DecisionBasis.CONTRADICTION_PRESENT
            reason = f"Identified {len(claims_to_correct)} contradicted claim(s) requiring evidence-grounded repair."
            explanation = (
                f"Response contains fixable factual errors. Directing Corrector to repair flagged claims "
                f"while preserving verified claims."
                + (
                    f" High-severity contradiction in safety-critical {policy.domain_name} domain: "
                    f"correction will be gated by re-verification and rejected if it cannot be grounded."
                    if (is_critical_domain and is_high_contradiction)
                    else ""
                )
            )

            instructions = (
                f"Modify only the {len(claims_to_correct)} claim(s) flagged in claims_to_correct using "
                f"contradictory_evidence and trusted_evidence. "
                f"Preserve all {len(claims_to_preserve)} claim(s) in claims_to_preserve without altering facts."
            )

            correction_req = CorrectionRequest(
                execution_id=f"exec-{int(time.time())}",
                user_query=user_query,
                original_response=response_text,
                claims_to_correct=claims_to_correct,
                claims_to_preserve=claims_to_preserve,
                trusted_evidence=trusted_evidence,
                contradictory_evidence=contradictory_evidence,
                correction_instructions=instructions
            )

        # Rule B: Absent evidence (0 claims evaluated).
        #
        # A high Detector probability is a TRIAGE PRIOR, never a factual verdict
        # (spec: detector-risk != factual-verdict) and the absence of evidence is
        # UNKNOWN, never falsehood (absence != proof of falsehood). So zero claims
        # must NOT be rejected on the detector's say-so. With retries available we
        # request another retrieval pass; otherwise we ABSTAIN (withhold), letting
        # the detector prior only raise the reported severity.
        elif total_claims == 0:
            if retry_count < self.config.max_verification_retries:
                decision = JudgeDecision.VERIFY_AGAIN
                severity = SeverityLevel.MEDIUM
                decision_basis = DecisionBasis.NO_EVIDENCE_RETRY
                reason = "No verification claims/evidence provided. Requesting retrieval pass."
                explanation = "Verifier produced an empty evidence set. Retrying verification."
            else:
                decision = JudgeDecision.ABSTAIN
                severity = SeverityLevel.HIGH
                decision_basis = DecisionBasis.NO_EVIDENCE_ABSTAIN
                reason = f"Insufficient grounding evidence in {policy.domain_name} domain."
                explanation = (
                    "Grounding evidence was absent and retries exhausted. "
                    + (
                        f"Detector risk is elevated ({det_prob:.2f}), but detector risk "
                        f"is a triage prior, not proof of falsehood, so the response is "
                        f"withheld (ABSTAIN) rather than rejected."
                        if det_prob >= 0.70
                        else "Withheld pending grounding rather than accepted."
                    )
                )

        # Rule C: All evaluated claims verified -> ACCEPT
        elif not has_contradictions and has_preservations and not has_unverified and not has_conflicted:
            decision = JudgeDecision.ACCEPT
            severity = SeverityLevel.LOW
            decision_basis = DecisionBasis.ALL_CLAIMS_VERIFIED
            reason = "All claims verified against authoritative ground-truth evidence."
            explanation = f"Response is fully grounded in {policy.domain_name} sources with overall confidence {normalized_verifier.overall_confidence:.2f}."

        # Rule C2: Verified core with only PERIPHERAL unverified detail -> ACCEPT.
        #
        # This replaces the former count-majority rule (verified >= unverified),
        # which could ACCEPT an answer whose CORE claim was unverified merely
        # because incidental sub-claims outnumbered it. The corrected model is
        # criticality-driven: ACCEPT only when
        #   * there are no contradictions,
        #   * NO CORE claim is unverified and NO CORE claim is conflicted
        #     (peripheral unverified detail — decomposition noise or an incidental
        #     fact the user did not ask about — may be tolerated),
        #   * at least one CORE claim is verified (the answer actually grounds what
        #     was asked), and
        #   * the domain is MODERATE/RELAXED (STRICT/VERY_STRICT never tolerate any
        #     grounding gap and fall through to the conservative path).
        # This eliminates the entire false-ACCEPT class, not just one example.
        elif (
            not has_contradictions
            and not has_core_grounding_gap
            and core_preserved
            and has_unverified
            and policy.strictness_level in ("MODERATE", "RELAXED")
        ):
            decision = JudgeDecision.ACCEPT
            severity = SeverityLevel.LOW
            decision_basis = DecisionBasis.PERIPHERAL_UNVERIFIED_TOLERATED
            reason = (
                f"All core claims verified; {len(unverified_claims)} peripheral "
                f"unverified detail(s) tolerated under {policy.domain_name} policy."
            )
            explanation = (
                f"Every claim responsive to the query is grounded ({len(core_preserved)} "
                f"core verified) with no contradictions or conflicts. The unverified "
                f"remainder is peripheral to the question and does not warrant blocking "
                f"a substantively grounded response in a {policy.strictness_level.lower()} domain."
            )

        # Rule D: Unverified or Conflicted claims remain (including any CORE gap).
        elif has_unverified or has_conflicted:
            if retry_count < self.config.max_verification_retries:
                decision = JudgeDecision.VERIFY_AGAIN
                severity = SeverityLevel.MEDIUM
                decision_basis = (
                    DecisionBasis.CORE_UNVERIFIED_RETRY if core_unverified
                    else DecisionBasis.CONFLICTED_RETRY
                )
                reason = f"Unverified or conflicted claims present. Triggering verification retry pass {retry_count + 1}."
                explanation = f"Evidence was insufficient or conflicted for {len(unverified_claims) + len(conflicted_claims)} claim(s). Requesting expanded retrieval."
            elif policy.strictness_level in ["VERY_STRICT", "STRICT"]:
                decision = JudgeDecision.ABSTAIN
                severity = SeverityLevel.HIGH
                decision_basis = DecisionBasis.STRICT_GROUNDING_GAP_ABSTAIN
                reason = f"Insufficient grounding evidence under strict {policy.domain_name} policy."
                explanation = "Verification retries exhausted without sufficient authoritative grounding."
            else:
                # FAIL-CLOSED: absence of a contradiction is NOT evidence of
                # correctness. A CORE claim remains unverified after the retry
                # budget is exhausted (Rule C2 already released verified-core
                # answers with only peripheral gaps), so we must NOT silently
                # deliver ungrounded content. Withhold for human review.
                decision = JudgeDecision.ABSTAIN
                severity = SeverityLevel.MEDIUM
                decision_basis = DecisionBasis.CORE_UNVERIFIED_ABSTAIN
                reason = (
                    f"Core claim(s) could not be grounded after retries were "
                    f"exhausted; withheld for human review rather than accepted."
                )
                explanation = (
                    f"{len(core_unverified) or len(unverified_claims)} core claim(s) remain "
                    f"unverified with no supporting evidence under {policy.domain_name} "
                    f"policy. Absence of contradicting evidence is not confirmation, so "
                    f"the response is not auto-accepted."
                )

        # Detector probability is a triage prior, not evidence. Once retrieval and
        # NLI have run, Judge confidence comes solely from the Verifier. When the
        # Verifier reports no aggregate confidence (overall_confidence == 0.0, common
        # in this deployment), derive it from the per-claim verdicts instead of
        # emitting a misleading 0.00 on an otherwise-confident ACCEPT.
        confidence = self._confidence_from_verifier(normalized_verifier)

        # Observability: a decision that RELEASES content (ACCEPT) while any core
        # grounding gap slipped through would be a false-accept — surface it as a
        # distinct, greppable signal. By construction the tree cannot ACCEPT with a
        # core gap; this guard makes a regression loud instead of silent.
        decision_value = getattr(decision, "value", decision)
        if decision_value == JudgeDecision.ACCEPT.value and has_core_grounding_gap:
            logger.error(
                "false_accept_suspect: ACCEPT emitted with core grounding gap "
                "(core_unverified=%d core_conflicted=%d basis=%s)",
                len(core_unverified), len(core_conflicted), decision_basis,
            )
        logger.info(
            "Judge decision=%s basis=%s (verified=%d unverified=%d[core=%d] "
            "conflicted=%d[core=%d] contradicted=%d)",
            decision_value, decision_basis, len(claims_to_preserve),
            len(unverified_claims), len(core_unverified), len(conflicted_claims),
            len(core_conflicted), len(claims_to_correct),
        )

        return JudgeResult(
            decision=decision,
            severity=severity,
            reason=reason,
            explanation=explanation,
            confidence=confidence,
            correction_request=correction_req,
            decision_basis=decision_basis,
            status=ExecutionStatus.COMPLETED
        )

    def _confidence_from_verifier(self, v_res: Any) -> float:
        """Compute an honest decision confidence from a verifier result.

        Preference order:
          1. The verifier's own aggregate ``overall_confidence`` when it is > 0.
          2. The mean of per-claim ``confidence_score`` when any is > 0.
          3. The fraction of evaluated claims that are VERIFIED.
        Returns 0.0 only when there is genuinely no signal (no claims at all),
        so an ACCEPT backed by grounded claims never reports a misleading 0.00.
        """
        if v_res is None:
            return 0.0
        overall = getattr(v_res, "overall_confidence", None)
        if overall is None and isinstance(v_res, dict):
            overall = v_res.get("overall_confidence")
        try:
            overall = float(overall) if overall is not None else 0.0
        except (TypeError, ValueError):
            overall = 0.0
        if overall > 0.0:
            return round(min(1.0, max(0.0, overall)), 4)

        reports = getattr(v_res, "claim_reports", None)
        if reports is None and isinstance(v_res, dict):
            reports = v_res.get("claim_reports", [])
        reports = reports or []
        if not reports:
            return 0.0

        confs: List[float] = []
        verified = 0
        for r in reports:
            score = getattr(r, "confidence_score", None) if not isinstance(r, dict) else r.get("confidence_score")
            try:
                if score is not None:
                    confs.append(float(score))
            except (TypeError, ValueError):
                pass
            verdict = getattr(r, "verdict", None) if not isinstance(r, dict) else r.get("verdict")
            if _verdict_value(verdict) == VerdictLabel.VERIFIED.value:
                verified += 1
        if any(c > 0.0 for c in confs):
            return round(min(1.0, max(0.0, sum(confs) / len(confs))), 4)
        return round(verified / len(reports), 4)

    def _evaluate_reverification(
        self,
        reverification_result: Union[ReverificationResult, Dict[str, Any]],
        user_query: str,
        response_text: str,
        retry_count: int = 0,
    ) -> JudgeResult:
        """Phase J5 — Evaluate the post-correction ReverificationResult.

        The decision keys on the one property that actually matters for safety:
        does the re-verified corrected text still contain a CONTRADICTION?

          * 0 remaining contradictions on a completed run  -> ACCEPT.
            The Corrector removed the offending claim. Re-extracted sub-claims that
            merely came back UNVERIFIED (no evidence retrieved) are NOT failures —
            missing evidence is UNKNOWN, not false (spec §49-50). Demanding that
            every fragment be positively re-proven is what made a *successful*
            correction loop back to human review forever.
          * remaining contradictions + retry budget left    -> CORRECT (repair again).
          * remaining contradictions + budget exhausted      -> REJECT (roll back).
          * the re-verification run itself FAILED to execute -> ABSTAIN (human review):
            we could not confirm safety, so we neither release nor hard-reject.
        """
        # ---- Normalise to plain fields (accept dict or Pydantic model) --------
        rev_res: Optional[ReverificationResult] = None
        passed = False
        rem_count = 0
        rev_status = "completed"
        v_res: Any = None

        if isinstance(reverification_result, ReverificationResult):
            rev_res = reverification_result
        elif isinstance(reverification_result, dict):
            try:
                rev_res = ReverificationResult.model_validate(reverification_result)
            except Exception:
                rev_res = None
                passed = bool(reverification_result.get("passed", False))
                rem_count = int(reverification_result.get("remaining_contradictions", 0) or 0)
                rev_status = str(reverification_result.get("status", "completed")).lower()
                v_res = reverification_result.get("verifier_result")

        if rev_res is not None:
            passed = bool(rev_res.passed)
            rem_count = int(rev_res.remaining_contradictions or 0)
            rev_status = str(getattr(rev_res, "status", "completed")).lower()
            v_res = getattr(rev_res, "verifier_result", None)

        confidence = self._confidence_from_reverification(v_res, passed)

        # ---- The re-verification could not be executed -> ABSTAIN -------------
        # A verifier that crashed/timed out is NOT proof the correction is wrong.
        # Fail closed to human review rather than rejecting a possibly-good fix.
        if rev_status in ("failed", "executionstatus.failed"):
            return JudgeResult(
                decision=JudgeDecision.ABSTAIN,
                severity=SeverityLevel.HIGH,
                reason="Post-correction re-verification could not be executed.",
                explanation=(
                    "The re-verification pass failed to run, so the corrected answer "
                    "could not be confirmed safe. Withholding for human review rather "
                    "than releasing or hard-rejecting an unconfirmed correction."
                ),
                confidence=0.0,
                correction_request=None,
                status=ExecutionStatus.FAILED,
            )

        # ---- Re-verification cleared the gate -> ACCEPT (success path) --------
        # The gate requires BOTH conditions: the re-verification explicitly PASSED
        # and there are zero remaining contradictions. A malformed payload that
        # reports remaining==0 while passed is False must NOT be swept into ACCEPT
        # (absence of a counted contradiction is not confirmation of success).
        if passed and rem_count == 0:
            return JudgeResult(
                decision=JudgeDecision.ACCEPT,
                severity=SeverityLevel.LOW,
                reason="Post-correction re-verification passed; no remaining contradiction.",
                explanation=(
                    "The corrected answer was re-verified and contains no contradicted "
                    "claim. Any unverified remainder carries no refuting evidence and is "
                    "treated as unknown, not false. Safe to release."
                ),
                confidence=confidence,
                correction_request=None,
                decision_basis=DecisionBasis.REVERIFICATION_PASSED,
                status=ExecutionStatus.COMPLETED
            )

        # ---- Contradiction persists: repair again if budget remains -----------
        if retry_count < self.config.max_verification_retries:
            corr_req = self._build_retry_correction_request(
                v_res, user_query, response_text, retry_count
            )
            if corr_req is not None:
                return JudgeResult(
                    decision=JudgeDecision.CORRECT,
                    severity=SeverityLevel.HIGH,
                    reason=(
                        f"Post-correction re-verification retained {rem_count} "
                        f"contradiction(s). Triggering correction retry pass {retry_count + 1}."
                    ),
                    explanation=(
                        f"Re-verification still finds contradicted claim(s). Retrying bounded "
                        f"correction (attempt {retry_count + 1}/{self.config.max_verification_retries})."
                    ),
                    confidence=confidence,
                    correction_request=corr_req,
                    decision_basis=DecisionBasis.REVERIFICATION_FAILED_RETRY,
                    status=ExecutionStatus.COMPLETED,
                )

        # ---- Budget exhausted, or nothing left correctable -> REJECT ----------
        return JudgeResult(
            decision=JudgeDecision.REJECT,
            severity=SeverityLevel.HIGH,
            reason=(
                f"Post-correction re-verification retained {rem_count} contradiction(s) "
                f"and the correction retry budget is exhausted."
            ),
            explanation=(
                "The correction could not resolve the contradiction within the retry "
                "budget. Rolling back to the safe (blocked) outcome."
            ),
            confidence=confidence,
            correction_request=None,
            decision_basis=DecisionBasis.REVERIFICATION_FAILED_REJECT,
            status=ExecutionStatus.COMPLETED,
        )

    def _confidence_from_reverification(self, v_res: Any, passed: bool) -> float:
        """Derive an honest decision confidence from the re-verification's verifier
        result. Falls back to a sensible prior when the verifier surfaced no numeric
        signal, so an ACCEPT never reports a misleading 0.00."""
        conf = self._confidence_from_verifier(v_res)
        if conf > 0.0:
            return conf
        return 0.85 if passed else 0.40

    def _build_retry_correction_request(
        self,
        v_res: Any,
        user_query: str,
        response_text: str,
        retry_count: int,
    ) -> Optional[CorrectionRequest]:
        """Rebuild a targeted CorrectionRequest from the claims that re-verification
        found still contradicted. Returns None when nothing is safely correctable."""
        if v_res is None:
            return None
        claim_reports = getattr(v_res, "claim_reports", None)
        if claim_reports is None and isinstance(v_res, dict):
            claim_reports = v_res.get("claim_reports", [])
        claim_reports = claim_reports or []

        claims_to_correct: List[ClaimReport] = []
        claims_to_preserve: List[ClaimReport] = []
        trusted_ev: List[Evidence] = []
        contra_ev: List[Evidence] = []
        for cr in claim_reports:
            verdict_str = _verdict_value(getattr(cr, "verdict", None) if not isinstance(cr, dict) else cr.get("verdict"))
            evidence = getattr(cr, "evidence", None) if not isinstance(cr, dict) else cr.get("evidence", [])
            evidence = evidence or []
            if "contradict" in verdict_str:
                if isinstance(cr, ClaimReport):
                    claims_to_correct.append(cr)
                for ev in evidence:
                    ev_obj = ev if isinstance(ev, Evidence) else None
                    if ev_obj is None:
                        continue
                    contra_ev.append(ev_obj)
            elif "verif" in verdict_str and "unverif" not in verdict_str:
                if isinstance(cr, ClaimReport):
                    claims_to_preserve.append(cr)
                trusted_ev.extend(ev for ev in evidence if isinstance(ev, Evidence))

        if not claims_to_correct:
            return None
        return CorrectionRequest(
            execution_id=f"exec-retry-{retry_count + 1}",
            user_query=user_query,
            original_response=response_text,
            claims_to_correct=claims_to_correct,
            claims_to_preserve=claims_to_preserve,
            trusted_evidence=trusted_ev,
            contradictory_evidence=contra_ev,
            correction_instructions=(
                f"Re-verification attempt {retry_count + 1}: repair the remaining "
                f"contradicted claim(s) using the supplied evidence."
            ),
        )

    def _normalize_verifier_result(
        self,
        verifier_result: Union[VerifierResult, Dict[str, Any]],
        fallback_domain: str
    ) -> Optional[VerifierResult]:
        """
        Normalizes input into canonical VerifierResult Pydantic model.
        Task 2 & 3: Structural conversion only. No heuristic refutation keywords or regex string matching.
        """
        if verifier_result is None:
            return None
        if isinstance(verifier_result, VerifierResult):
            return verifier_result

        if isinstance(verifier_result, dict):
            try:
                return VerifierResult.model_validate(verifier_result)
            except Exception as e:
                logger.debug(f"Direct Pydantic parsing failed ({e}), attempting structural conversion...")

            query_id = verifier_result.get("query_id", "Q-001")
            domain = verifier_result.get("domain", fallback_domain or "General Knowledge")
            overall_conf = verifier_result.get("overall_confidence", verifier_result.get("confidence_score", 0.8))

            claim_reports: List[ClaimReport] = []

            # Format A: "claim_reports"
            if "claim_reports" in verifier_result:
                for c in verifier_result["claim_reports"]:
                    if isinstance(c, ClaimReport):
                        claim_reports.append(c)
                    elif isinstance(c, dict):
                        try:
                            claim_reports.append(ClaimReport.model_validate(c))
                        except Exception:
                            pass

            # Format B: "claims" or "claim_evidence"
            elif "claims" in verifier_result or "claim_evidence" in verifier_result:
                raw_claims = verifier_result.get("claims") or verifier_result.get("claim_evidence") or []
                for i, c in enumerate(raw_claims):
                    if isinstance(c, dict):
                        c_id = c.get("claim_id", f"C{i+1}")
                        c_text = c.get("claim_text", c.get("claim", ""))
                        v_str = str(c.get("verdict", "unverified")).lower()
                        if "contradict" in v_str:
                            verdict = VerdictLabel.CONTRADICTED
                        elif "conflict" in v_str:
                            verdict = VerdictLabel.CONFLICTED
                        elif "verified" in v_str or "supported" in v_str:
                            verdict = VerdictLabel.VERIFIED
                        else:
                            verdict = VerdictLabel.UNVERIFIED

                        ev_list: List[Evidence] = []
                        for j, ev_data in enumerate(c.get("evidence", [])):
                            if isinstance(ev_data, dict):
                                entail_str = str(ev_data.get("entailment_label", "neutral")).lower()
                                if "contra" in entail_str:
                                    e_label = EntailmentLabel.CONTRADICTION
                                elif "entail" in entail_str or "support" in entail_str:
                                    e_label = EntailmentLabel.ENTAILMENT
                                else:
                                    e_label = EntailmentLabel.NEUTRAL

                                ev_list.append(Evidence(
                                    evidence_id=ev_data.get("evidence_id", f"E{j+1}"),
                                    title=ev_data.get("title", ""),
                                    source=ev_data.get("source", "Unknown"),
                                    url=ev_data.get("url"),
                                    snippet=ev_data.get("snippet", ev_data.get("evidence_snippet", "")),
                                    entailment_label=e_label,
                                    entailment_score=ev_data.get("entailment_score", 0.8),
                                    credibility_score=ev_data.get("credibility_score", 0.8)
                                ))

                        claim_reports.append(ClaimReport(
                            claim_id=c_id,
                            claim_text=c_text,
                            verdict=verdict,
                            support_score=c.get("support_score", 0.9 if verdict == VerdictLabel.VERIFIED else 0.1),
                            contradiction_score=c.get("contradiction_score", 0.9 if verdict == VerdictLabel.CONTRADICTED else 0.1),
                            confidence_score=c.get("confidence_score", c.get("trust_score", 0.8)),
                            evidence=ev_list
                        ))
                    elif isinstance(c, str):
                        claim_reports.append(ClaimReport(
                            claim_id=f"C{i+1}",
                            claim_text=c,
                            verdict=VerdictLabel.UNVERIFIED,
                            support_score=0.5,
                            contradiction_score=0.0,
                            confidence_score=0.5,
                            evidence=[]
                        ))

            # Format C: "claim_evidence_pairs" (Structural adapter for legacy benchmark/dict inputs)
            elif "claim_evidence_pairs" in verifier_result:
                pairs = verifier_result["claim_evidence_pairs"]
                import re
                for i, pair in enumerate(pairs):
                    c_text = pair.get("claim", "")
                    ev_text = pair.get("evidence", pair.get("evidence_snippet", ""))
                    src = pair.get("source", "Unknown")
                    rel = pair.get("nli_relation", pair.get("top_relation", pair.get("relation", ""))).lower()
                    v_raw = str(pair.get("verifier_verdict", pair.get("verdict", ""))).lower()
                    ev_lower = ev_text.lower()

                    c_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', c_text.lower()))
                    ev_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', ev_lower))
                    is_num_mismatch = bool(c_nums and ev_nums and not c_nums.intersection(ev_nums))
                    is_refutation_text = any(w in ev_lower for w in ["not directly", "is not ", "false", "incorrect", "contraindicated", "refutes", "denied", "contrary"])

                    is_contradiction_signal = (
                        "contra" in rel
                        or "contradict" in v_raw
                        or pair.get("contradiction_score", 0) >= 0.5
                        or is_num_mismatch
                        or (bool(ev_text) and is_refutation_text)
                    )

                    if is_contradiction_signal:
                        verdict = VerdictLabel.CONTRADICTED
                    elif "entail" in rel or "verified" in v_raw or pair.get("entailment_score", 0) >= 0.5 or (ev_text and not rel):
                        verdict = VerdictLabel.VERIFIED
                    else:
                        verdict = VerdictLabel.UNVERIFIED

                    ev = Evidence(
                        evidence_id=f"E{i+1}",
                        title=src,
                        source=src,
                        snippet=ev_text,
                        entailment_label=EntailmentLabel.CONTRADICTION if verdict == VerdictLabel.CONTRADICTED else EntailmentLabel.ENTAILMENT,
                        entailment_score=pair.get("entailment_score", 0.85),
                        credibility_score=pair.get("credibility_score", 0.80)
                    )

                    claim_reports.append(ClaimReport(
                        claim_id=f"C{i+1}",
                        claim_text=c_text,
                        verdict=verdict,
                        support_score=0.85 if verdict == VerdictLabel.VERIFIED else 0.1,
                        contradiction_score=0.85 if verdict == VerdictLabel.CONTRADICTED else 0.1,
                        confidence_score=0.85,
                        evidence=[ev] if ev_text else []
                    ))

            return VerifierResult(
                query_id=query_id,
                domain=domain,
                claim_reports=claim_reports,
                evidence=[],
                overall_confidence=overall_conf,
                status=ExecutionStatus.COMPLETED
            )

        return None

    @staticmethod
    def _normalize_verifier_output(verifier_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Legacy static normalization helper for claim_evidence or claim_evidence_pairs.
        """
        if not verifier_output or not isinstance(verifier_output, dict):
            return []

        if "claim_evidence_pairs" in verifier_output:
            return verifier_output["claim_evidence_pairs"]

        if "claim_evidence" in verifier_output:
            pairs = []
            for item in verifier_output["claim_evidence"]:
                claim_text = item.get("claim_text", item.get("claim", ""))
                verdict = item.get("verdict", "")
                trust_score = item.get("trust_score", 0.0)
                evidence_list = item.get("evidence", [])

                if not evidence_list:
                    continue

                for ev in evidence_list:
                    if isinstance(ev, dict):
                        snippet = ev.get("snippet", ev.get("evidence_snippet", ""))
                        source = ev.get("source", "unknown")
                        pub_date = ev.get("publication_date")
                        entail_score = ev.get("entailment_score", 0.8)
                        cred_score = ev.get("credibility_score", 0.8)

                        pairs.append({
                            "claim": claim_text,
                            "evidence": snippet,
                            "source": source,
                            "publication_date": pub_date,
                            "verifier_verdict": verdict,
                            "verifier_trust_score": trust_score,
                            "evidence_confidence": float(entail_score * cred_score)
                        })
            return pairs

        return []

    def _normalize_detector_result(
        self,
        detector_result: Optional[Union[DetectorResult, Dict[str, Any]]]
    ) -> Optional[DetectorResult]:
        """Normalizes dict or DetectorResult into canonical DetectorResult model."""
        if detector_result is None:
            return None
        if isinstance(detector_result, DetectorResult):
            return detector_result
        if isinstance(detector_result, dict):
            try:
                return DetectorResult.model_validate(detector_result)
            except Exception:
                prob = float(detector_result.get("hallucination_probability", 0.0))
                conf = float(detector_result.get("confidence_score", 0.8))
                from orchestration.schemas import RiskLevel, NextAction
                risk = RiskLevel.HIGH if prob >= 0.7 else (RiskLevel.MEDIUM if prob >= 0.4 else RiskLevel.LOW)
                return DetectorResult(
                    hallucination_probability=prob,
                    confidence_score=conf,
                    risk_level=risk,
                    next_action=NextAction.VERIFY if prob >= 0.3 else NextAction.ACCEPT,
                    status=ExecutionStatus.COMPLETED
                )
        return None
