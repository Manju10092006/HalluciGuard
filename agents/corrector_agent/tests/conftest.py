"""Shared pytest fixtures for the Corrector Agent test suite.

Inserts the repository root on ``sys.path`` (mirroring
``orchestration/tests/test_canonical_contracts.py``) so the suite runs whether
invoked via the root ``pytest.ini`` (``pythonpath = .``) or standalone by path.

Step 5 additions live at the bottom and are purely additive: the Step 1 fixtures
above them are untouched.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

# tests/ -> corrector_agent/ -> agents/ -> <repo root>
ROOT_DIR = str(Path(__file__).resolve().parents[3])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest

from orchestration.schemas import (
    ClaimReport,
    CorrectionRequest,
    Evidence,
    VerdictLabel,
)


@pytest.fixture
def trusted_evidence_item() -> Evidence:
    return Evidence(
        evidence_id="ev-guido",
        title="Python (programming language)",
        source="wikipedia",
        url="https://en.wikipedia.org/wiki/Python_(programming_language)",
        snippet="Python was created by Guido van Rossum and first released in 1991.",
        entailment_label="contradiction",
        entailment_score=0.97,
        credibility_score=0.90,
    )


@pytest.fixture
def correct_claim(trusted_evidence_item: Evidence) -> ClaimReport:
    return ClaimReport(
        claim_id="c2",
        claim_text="Python was created by Elon Musk.",
        verdict=VerdictLabel.CONTRADICTED,
        support_score=0.02,
        contradiction_score=0.98,
        confidence_score=0.95,
        evidence=[trusted_evidence_item],
    )


@pytest.fixture
def preserve_claim() -> ClaimReport:
    return ClaimReport(
        claim_id="c1",
        claim_text="The sky is blue.",
        verdict=VerdictLabel.VERIFIED,
        support_score=0.99,
        contradiction_score=0.0,
        confidence_score=0.97,
        evidence=[],
    )


@pytest.fixture
def valid_correction_request(
    correct_claim: ClaimReport,
    preserve_claim: ClaimReport,
    trusted_evidence_item: Evidence,
) -> CorrectionRequest:
    return CorrectionRequest(
        execution_id="exec-123",
        user_query="Tell me about Python and the sky.",
        original_response="The sky is blue. Python was created by Elon Musk.",
        claims_to_correct=[correct_claim],
        claims_to_preserve=[preserve_claim],
        trusted_evidence=[trusted_evidence_item],
        contradictory_evidence=[],
        correction_instructions="Preserve c1 exactly. Rewrite c2 using ev-guido.",
    )


# ===========================================================================
# Step 5 fixtures
# ===========================================================================
#
# Step 5 tests are built on REAL Step 1-4 artefacts: every plan, prompt bundle and
# candidate below comes from ``to_internal_request`` -> ``ground_request`` ->
# ``build_prompts`` -> ``parse_candidate``. Nothing is hand-constructed.
#
# That matters because the properties Step 5 relies on -- that
# ``evidence_ids_used`` is the render manifest and not model output, that
# ``original_text`` is a byte-exact slice of the response, that only supporting
# evidence is ever bound -- are guarantees of those upstream functions. A
# hand-built fixture would silently assume them and the tests would stop proving
# anything about the real pipeline.

from agents.corrector_agent.corrector import output_parser as _op  # noqa: E402
from agents.corrector_agent.corrector import prompt_builder as _pb  # noqa: E402
from agents.corrector_agent.corrector.adapter import (  # noqa: E402
    to_internal_request as _to_internal_request,
)
from agents.corrector_agent.corrector.evidence import (  # noqa: E402
    ground_request as _ground_request,
)
from agents.corrector_agent.corrector.validation import (  # noqa: E402
    build_context as _build_context,
)

# ---------------------------------------------------------------------------
# The recorded echo failure
# ---------------------------------------------------------------------------
#
# Verbatim from ``training/data/contract_v4/test_benchmark_report.json``
# (``detailed_sample_results[0]``) and the matching first record of
# ``test.jsonl``. This is the real observed failure of the fine-tuned model, not
# an invented example, which is why the mandatory test uses these exact strings.
#
# Note what the recorded output does: it echoes the original and ADDS a full stop.
# So it is NOT byte-identical to the original, and an equality-only echo check
# misses it. This single record is the reason the echo ladder has more than one
# rung.

RECORDED_QUERY = "How long should you wait before filing a missing person report?"
RECORDED_ORIGINAL = "You should wait 48 hours before filing a missing person report"
RECORDED_EVIDENCE = (
    "You can make a missing person report as soon as you think a person is missing"
)
RECORDED_GOLD = (
    "You can make a missing person report as soon as you think a person is missing"
)
RECORDED_MODEL_OUTPUT = (
    "You should wait 48 hours before filing a missing person report."
)
RECORDED_SOURCE = (
    "https://en.wikipedia.org/wiki/List_of_common_misconceptions"
    "#Law,_crime,_and_military"
)

# The canonical multi-sentence response used by the general-purpose cases. S1 is
# preserved, S2 is the correction target, S3 exists so "did the model rewrite a
# sentence it was not authorized to touch" is testable.
STEP5_RESPONSE = (
    "The sky is blue. Python was created by Elon Musk in 1823. Paris is in Spain."
)
STEP5_CLAIM_TEXT = "Python was created by Elon Musk in 1823."
STEP5_EVIDENCE = "Python was created by Guido van Rossum in 1991."
STEP5_GOOD_CORRECTION = "Python was created by Guido van Rossum in 1991."
STEP5_ABSTENTION = "Current evidence is insufficient to support this claim."


@dataclass(frozen=True)
class Step5Case:
    """One fully-grounded single-target scenario, ready to validate against."""

    internal: Any
    plan: Any
    bundle: Any
    context: Any
    grounded: Any

    @property
    def sentence_id(self) -> str:
        return self.bundle.sentence_id

    @property
    def original_text(self) -> str:
        return self.grounded.target.original_text


class Step5Kit:
    """Builders for Step 5 tests. Every artefact is produced by real Step 1-4 code."""

    response = STEP5_RESPONSE
    claim_text = STEP5_CLAIM_TEXT
    evidence_text = STEP5_EVIDENCE
    good = STEP5_GOOD_CORRECTION
    abstention = STEP5_ABSTENTION

    # The recorded missing-person failure, exposed so tests can reference the real
    # observed strings without importing conftest as a module. Single source of
    # truth: these alias the module-level RECORDED_* constants above.
    recorded_query = RECORDED_QUERY
    recorded_original = RECORDED_ORIGINAL
    recorded_evidence = RECORDED_EVIDENCE
    recorded_output = RECORDED_MODEL_OUTPUT

    # -- canonical inputs -------------------------------------------------

    def evidence(
        self,
        evidence_id: str = "ev-1",
        snippet: str = STEP5_EVIDENCE,
        label: str = "entailment",
        title: str = "Python history",
        source: str = "example.org",
        url: Optional[str] = "https://example.org/python",
        entailment_score: float = 0.91,
        credibility_score: float = 0.83,
    ) -> Evidence:
        return Evidence(
            evidence_id=evidence_id,
            title=title,
            source=source,
            url=url,
            snippet=snippet,
            entailment_label=label,
            entailment_score=entailment_score,
            credibility_score=credibility_score,
        )

    def claim(
        self,
        claim_id: str = "c1",
        text: str = STEP5_CLAIM_TEXT,
        evidence: Optional[Sequence[Evidence]] = None,
        verdict: str = "contradicted",
    ) -> ClaimReport:
        return ClaimReport(
            claim_id=claim_id,
            claim_text=text,
            verdict=verdict,
            evidence=list(evidence or []),
        )

    def request(
        self,
        correct: Sequence[ClaimReport] = (),
        preserve: Sequence[ClaimReport] = (),
        trusted: Sequence[Evidence] = (),
        contradictory: Sequence[Evidence] = (),
        original: str = STEP5_RESPONSE,
        instructions: str = "",
        query: str = "Who created Python?",
    ) -> CorrectionRequest:
        return CorrectionRequest(
            execution_id="exec-step5",
            user_query=query,
            original_response=original,
            claims_to_correct=list(correct),
            claims_to_preserve=list(preserve),
            trusted_evidence=list(trusted),
            contradictory_evidence=list(contradictory),
            correction_instructions=instructions,
        )

    # -- real pipeline ----------------------------------------------------

    def ground(self, request: CorrectionRequest):
        internal = _to_internal_request(request)
        return internal, _ground_request(internal)

    def bundles(self, internal, plan) -> List[Any]:
        return _pb.build_prompts(internal, plan)

    def case(
        self,
        *,
        claim_text: str = STEP5_CLAIM_TEXT,
        snippets: Sequence[str] = (STEP5_EVIDENCE,),
        labels: Optional[Sequence[str]] = None,
        response: str = STEP5_RESPONSE,
        query: str = "Who created Python?",
        instructions: str = "",
        preserve: Sequence[ClaimReport] = (),
        index: int = 0,
    ) -> Step5Case:
        """Build a single-target case with one claim and the given evidence."""
        marks = list(labels or ["entailment"] * len(snippets))
        evidence = [
            self.evidence(f"ev-{i}", snippet, label=mark)
            for i, (snippet, mark) in enumerate(zip(snippets, marks), start=1)
        ]
        claim = self.claim("c1", claim_text, evidence)
        request = self.request(
            correct=[claim],
            preserve=preserve,
            original=response,
            query=query,
            instructions=instructions,
        )
        internal, plan = self.ground(request)
        grounded = plan.grounded_targets[index]
        bundles = [b for b in self.bundles(internal, plan) if b.sentence_id ==
                   grounded.target.sentence_id]
        bundle = bundles[0] if bundles else None
        context = _build_context(
            target=grounded, plan=plan, bundle=bundle, query_text=internal.user_query
        )
        return Step5Case(internal, plan, bundle, context, grounded)

    def two_target_case(self):
        """Two independent authorized targets, for cross-target isolation tests."""
        c1 = self.claim("c1", STEP5_CLAIM_TEXT, [self.evidence("ev-guido")])
        c2 = self.claim(
            "c2",
            "Paris is in Spain.",
            [
                self.evidence(
                    "ev-paris",
                    "Paris is the capital of France.",
                    title="Geography",
                    source="atlas.example",
                    url="https://atlas.example/paris",
                )
            ],
        )
        internal, plan = self.ground(self.request(correct=[c1, c2]))
        return internal, plan, self.bundles(internal, plan)

    def recorded_case(self) -> Step5Case:
        """The recorded missing-person echo failure, as a validatable case."""
        evidence = self.evidence(
            "ev-1",
            RECORDED_EVIDENCE,
            title="Verified Ground Truth",
            source=RECORDED_SOURCE,
            url=RECORDED_SOURCE,
            entailment_score=0.96,
            credibility_score=0.92,
        )
        claim = self.claim("c1", RECORDED_ORIGINAL, [evidence])
        request = self.request(
            correct=[claim], original=RECORDED_ORIGINAL, query=RECORDED_QUERY
        )
        internal, plan = self.ground(request)
        grounded = plan.grounded_targets[0]
        bundle = self.bundles(internal, plan)[0]
        context = _build_context(
            target=grounded, plan=plan, bundle=bundle, query_text=internal.user_query
        )
        return Step5Case(internal, plan, bundle, context, grounded)

    # -- candidates -------------------------------------------------------

    def candidate(self, bundle, text: str, sentence_id: Optional[str] = None):
        """A candidate produced by the REAL parser, so its provenance is real."""
        payload = json.dumps(
            {
                "sentence_id": sentence_id or bundle.sentence_id,
                "corrected_sentence": text,
            }
        )
        return _op.parse_candidate(payload, bundle)

    def raw_candidate(self, bundle, text: str, **overrides):
        """A candidate with fields overridden — for states the parser cannot emit."""
        return self.candidate(bundle, text).model_copy(update=overrides)


@pytest.fixture
def step5() -> Step5Kit:
    return Step5Kit()


@pytest.fixture
def step5_case(step5: Step5Kit) -> Step5Case:
    return step5.case()


@pytest.fixture
def recorded_echo_case(step5: Step5Kit) -> Step5Case:
    return step5.recorded_case()
