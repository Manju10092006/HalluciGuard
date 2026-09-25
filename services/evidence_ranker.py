"""Deterministic multi-signal evidence relevance ranking.

The Snehith/Google failure was caused by treating token overlap
(``adapter_score``) as the primary relevance signal: a Wikipedia page about an
unrelated actor named "Snehith" out-scored real counter-evidence about Google's
actual founders. This module replaces single-signal overlap with a bundle of
independent deterministic signals, then buckets each passage into a tier so only
genuinely relevant evidence reaches NLI / the Verifier.

Design contract:

* No network, no model — pure Python over the passage text + metadata, so it is
  fully unit-testable offline and cheap enough to run on every retrieval.
* Supports COUNTER-EVIDENCE: a passage that refutes a claim is relevant even if
  it does not repeat the claim's subject ("Google was founded by Larry Page and
  Sergey Brin" is relevant to "Snehith is the founder of Google").
* Ranking is a weighted sum of signals (§18); tiers are STRONG / USABLE / WEAK /
  DISCARD (§19). Only STRONG + selected USABLE are returned, top-k per claim,
  with source-family diversity so five Wikipedia hits do not crowd out an
  independent source.
* Every discarded passage carries a machine-readable discard reason (§34).

It never decides truth. Contradiction vs. support is the NLI / Verifier's job;
this module only decides *relevance*.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import urlparse

logger = logging.getLogger("HalluciGuard.EvidenceRanker")

# Tier labels (§19).
STRONG = "STRONG"
USABLE = "USABLE"
WEAK = "WEAK"
DISCARD = "DISCARD"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


# Ranking weights (§18). Configurable per deployment; must sum to ~1.0.
WEIGHTS = {
    "relevance": _env_float("HG_RANK_W_RELEVANCE", 0.30),
    "entity": _env_float("HG_RANK_W_ENTITY", 0.20),
    "authority": _env_float("HG_RANK_W_AUTHORITY", 0.15),
    "coverage": _env_float("HG_RANK_W_COVERAGE", 0.10),
    "specificity": _env_float("HG_RANK_W_SPECIFICITY", 0.10),
    "temporal": _env_float("HG_RANK_W_TEMPORAL", 0.05),
    "domain": _env_float("HG_RANK_W_DOMAIN", 0.05),
    "diversity": _env_float("HG_RANK_W_DIVERSITY", 0.05),
}

# Tier thresholds (§19).
TIER_STRONG = _env_float("HG_TIER_STRONG", 0.75)
TIER_USABLE = _env_float("HG_TIER_USABLE", 0.55)
TIER_WEAK = _env_float("HG_TIER_WEAK", 0.35)

# Selection limits.
TOP_K = _env_int("HG_RANK_TOP_K", 5)
MIN_K = _env_int("HG_RANK_MIN_K", 3)
MAX_PER_FAMILY = _env_int("HG_RANK_MAX_PER_FAMILY", 2)

# Configurable source-authority priors, matched by host substring. Domain-aware
# priors can be layered by callers; these are conservative general defaults.
AUTHORITY_PRIORS: dict[str, float] = {
    "wikipedia.org": 0.80,
    "britannica.com": 0.85,
    ".gov": 0.95,
    ".edu": 0.85,
    "who.int": 0.95,
    "nih.gov": 0.95,
    "ncbi.nlm.nih.gov": 0.95,
    "pubmed": 0.95,
    "europepmc.org": 0.90,
    "clinicaltrials.gov": 0.95,
    "fda.gov": 0.95,
    "nvd.nist.gov": 0.95,
    "osv.dev": 0.90,
    "arxiv.org": 0.75,
    "semanticscholar.org": 0.80,
    "dblp.org": 0.80,
    "sec.gov": 0.95,
    "courtlistener.com": 0.90,
    "usgs.gov": 0.95,
    "pubchem.ncbi.nlm.nih.gov": 0.95,
    "uniprot.org": 0.90,
    "pypi.org": 0.75,
    "archive.org": 0.65,
}
DEFAULT_AUTHORITY = _env_float("HG_DEFAULT_AUTHORITY", 0.45)

_STOPWORDS = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "with", "and",
    "or", "is", "are", "was", "were", "be", "been", "being", "who", "what",
    "when", "where", "which", "why", "how", "that", "this", "these", "those",
    "it", "its", "as", "from", "about", "into", "than", "then", "there",
    "has", "have", "had", "not", "no", "also", "but", "his", "her", "their",
})

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']*")
_CAP_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]+)\b")
_NUM_RE = re.compile(r"\b\d{2,}\b")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _content_terms(text: str) -> set[str]:
    out: set[str] = set()
    for tok in _tokens(text):
        if tok.isdigit():
            out.add(tok)
        elif len(tok) > 2 and tok not in _STOPWORDS:
            out.add(tok)
    return out


def _entities(text: str) -> set[str]:
    """Approximate named entities: capitalized words + multi-digit numbers.

    Deterministic and dependency-free. Lower-cased for comparison; the first
    word of a sentence adds mild noise but the overlap metric tolerates it.
    """
    ents = {m.group(1).lower() for m in _CAP_RE.finditer(text or "")}
    ents |= {m.group(0) for m in _NUM_RE.finditer(text or "")}
    return {e for e in ents if e not in _STOPWORDS and len(e) > 1}


def _field(obj: Any, name: str, default: Any = "") -> Any:
    """Read a field from a Passage-like object or a dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _passage_text(passage: Any) -> str:
    title = str(_field(passage, "title", "") or "")
    snippet = str(_field(passage, "snippet", "") or _field(passage, "text", "") or "")
    return f"{title}. {snippet}".strip()


def source_family(passage: Any) -> str:
    """Group key for source diversity: registered domain (5 Wikipedia = 1 family)."""
    url = str(_field(passage, "url", "") or "")
    host = urlparse(url).netloc.lower() if url else ""
    if host.startswith("www."):
        host = host[4:]
    if host:
        parts = host.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return host
    src = str(_field(passage, "source", "") or _field(passage, "source_id", "") or "")
    return src.strip().lower() or "unknown"


def _authority(passage: Any) -> float:
    url = str(_field(passage, "url", "") or "").lower()
    source = str(_field(passage, "source", "") or "").lower()
    hay = f"{url} {source}"
    best = 0.0
    for key, prior in AUTHORITY_PRIORS.items():
        if key in hay:
            best = max(best, prior)
    return best if best > 0 else DEFAULT_AUTHORITY


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _coverage(claim_terms: set[str], passage_terms: set[str]) -> float:
    """Fraction of the claim's content terms present in the passage."""
    if not claim_terms:
        return 0.0
    return len(claim_terms & passage_terms) / len(claim_terms)


def _overlap_coeff(a: set[str], b: set[str]) -> float:
    """Szymkiewicz-Simpson overlap: intersection / size of the smaller set.

    Used for entity relevance so counter-evidence (which shares the object /
    predicate entity but not the — often false — subject) is not diluted by the
    passage's many additional entities the way symmetric Jaccard would.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _specificity(text: str) -> float:
    """How concrete the passage is: presence of numbers, dates, proper nouns."""
    score = 0.0
    if _NUM_RE.search(text):
        score += 0.5
    caps = len(set(_CAP_RE.findall(text)))
    score += min(0.5, caps * 0.1)
    return min(1.0, score)


def _temporal(passage: Any) -> float:
    """Recency prior in [0,1]. Neutral 0.5 when no date is available."""
    raw = str(_field(passage, "publication_date", "") or "").strip()
    if not raw:
        return 0.5
    year = None
    m = re.search(r"(19|20)\d{2}", raw)
    if m:
        year = int(m.group(0))
    if year is None:
        return 0.5
    now = datetime.now(timezone.utc).year
    age = max(0, now - year)
    # Full credit within 3 years, linear decay to 0.3 over 25 years.
    if age <= 3:
        return 1.0
    return max(0.3, 1.0 - (age - 3) * (0.7 / 22))


@dataclass
class RankedEvidence:
    """A passage plus its relevance verdict and per-signal breakdown."""

    passage: Any
    score: float
    tier: str
    signals: dict[str, float] = field(default_factory=dict)
    family: str = ""
    discarded: bool = False
    discard_reason: str = ""
    rank: int = 0

    def observability(self) -> dict[str, Any]:
        return {
            "url": str(_field(self.passage, "url", "") or ""),
            "title": str(_field(self.passage, "title", "") or ""),
            "family": self.family,
            "rank": self.rank,
            "score": round(self.score, 4),
            "tier": self.tier,
            "discarded": self.discarded,
            "discard_reason": self.discard_reason,
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
        }


_SUFFIXES = ("ers", "ors", "ing", "ed", "es", "s", "er", "or")


def _stem(term: str) -> str:
    """Very light suffix stripper so founder/founded/founders/founding collapse.

    Deliberately crude and deterministic — enough to align predicate variants
    ("founder" vs "founded") that token overlap would otherwise miss, which is
    exactly what let counter-evidence rank below a same-subject false match.
    """
    t = term
    for suf in _SUFFIXES:
        if len(t) > len(suf) + 2 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def _stems(terms: Iterable[str]) -> set[str]:
    return {_stem(t) for t in terms}


def score_passage(
    passage: Any,
    *,
    claim_text: str,
    search_queries: Iterable[str] = (),
    domain: str = "general",
) -> RankedEvidence:
    """Score one passage against a claim across all relevance signals (§18)."""
    text = _passage_text(passage)
    p_terms = _stems(_content_terms(text))
    p_ents = _stems(_entities(text))

    query_blob = " ".join([claim_text, *[q for q in search_queries if q]])
    c_terms = _stems(_content_terms(claim_text))
    q_terms = _stems(_content_terms(query_blob))
    c_ents = _stems(_entities(claim_text))

    # Query relevance: how much of the claim + its search-query anchors the
    # passage covers. Coverage-oriented (not symmetric Jaccard) so relevant
    # counter-evidence is not penalized for the claim's absent (false) subject.
    relevance = _coverage(q_terms, p_terms)
    # Entity relevance: shared salient entities via overlap coefficient, so a
    # passage sharing the object/predicate entity scores well even when it
    # carries many other entities of its own.
    entity = _overlap_coeff(c_ents, p_ents) if c_ents else relevance
    # Claim coverage: fraction of claim content terms the passage covers. This is
    # what lets predicate+object counter-evidence beat a bare same-subject match.
    coverage = _coverage(c_terms, p_terms)
    authority = _authority(passage)
    specificity = _specificity(text)
    temporal = _temporal(passage)
    domain_match = 1.0 if authority >= 0.80 else (0.6 if authority >= 0.6 else 0.4)

    # Blend in the upstream retrieval score (adapter/source confidence hint) as a
    # soft floor on relevance so a strong retriever match is not thrown away, but
    # never as the primary signal (that was the original bug).
    retrieval_hint = 0.0
    try:
        retrieval_hint = float(_field(passage, "source_confidence_hint", 0.0) or 0.0)
    except (TypeError, ValueError):
        retrieval_hint = 0.0
    retrieval_hint = max(0.0, min(1.0, retrieval_hint))
    relevance = max(relevance, 0.5 * retrieval_hint * min(1.0, coverage + 0.34))

    signals = {
        "relevance": relevance,
        "entity": entity,
        "authority": authority,
        "coverage": coverage,
        "specificity": specificity,
        "temporal": temporal,
        "domain": domain_match,
        "retrieval_hint": retrieval_hint,
    }
    # Diversity is resolved at selection time; per-passage it contributes its full
    # weight so a lone passage is not penalized for being alone.
    base = (
        WEIGHTS["relevance"] * relevance
        + WEIGHTS["entity"] * entity
        + WEIGHTS["authority"] * authority
        + WEIGHTS["coverage"] * coverage
        + WEIGHTS["specificity"] * specificity
        + WEIGHTS["temporal"] * temporal
        + WEIGHTS["domain"] * domain_match
        + WEIGHTS["diversity"] * 1.0
    )
    score = max(0.0, min(1.0, base))
    return RankedEvidence(
        passage=passage,
        score=score,
        tier=_tier_for(score),
        signals=signals,
        family=source_family(passage),
    )


def _tier_for(score: float) -> str:
    if score >= TIER_STRONG:
        return STRONG
    if score >= TIER_USABLE:
        return USABLE
    if score >= TIER_WEAK:
        return WEAK
    return DISCARD


def _canonical_url(passage: Any) -> str:
    url = str(_field(passage, "url", "") or "").strip().lower()
    if not url:
        return ""
    parsed = urlparse(url)
    base = f"{parsed.netloc}{parsed.path}".rstrip("/")
    # A section/chunk URL is not a duplicate of the article lead. Retrieval
    # adapters deliberately emit deep Wikipedia sections because the lead often
    # omits the exact relation under verification (creator, parent, release
    # date, etc.). Dropping the fragment here removed the strongest evidence
    # before BGE/NLI — e.g. Rust#2006-2009_Early_years was collapsed into the
    # generic Rust article whose lead does not mention Graydon Hoare.
    fragment = parsed.fragment.strip().lower()
    return f"{base}#{fragment}" if fragment else base


def _dedup(passages: list[Any]) -> tuple[list[Any], list[tuple[Any, str]]]:
    """Drop URL / canonical-URL / near-duplicate content passages.

    Returns (kept, dropped_with_reason).
    """
    kept: list[Any] = []
    dropped: list[tuple[Any, str]] = []
    seen_urls: set[str] = set()
    seen_canon: set[str] = set()
    kept_termsets: list[set[str]] = []
    for p in passages:
        url = str(_field(p, "url", "") or "").strip().lower()
        canon = _canonical_url(p)
        if url and url in seen_urls:
            dropped.append((p, "duplicate_url"))
            continue
        if canon and canon in seen_canon:
            dropped.append((p, "duplicate_canonical_url"))
            continue
        terms = _stems(_content_terms(_passage_text(p)))
        if any(_jaccard(terms, t) >= 0.9 for t in kept_termsets if t):
            dropped.append((p, "duplicate_content"))
            continue
        if url:
            seen_urls.add(url)
        if canon:
            seen_canon.add(canon)
        kept_termsets.append(terms)
        kept.append(p)
    return kept, dropped


def rank_evidence(
    passages: Iterable[Any],
    *,
    claim_text: str,
    search_queries: Iterable[str] = (),
    domain: str = "general",
) -> list[RankedEvidence]:
    """Dedup + score every passage, returned sorted by score (desc)."""
    plist = list(passages)
    kept, dropped = _dedup(plist)
    ranked = [
        score_passage(p, claim_text=claim_text, search_queries=search_queries, domain=domain)
        for p in kept
    ]
    for p, reason in dropped:
        re_item = RankedEvidence(
            passage=p, score=0.0, tier=DISCARD, family=source_family(p),
            discarded=True, discard_reason=reason,
        )
        ranked.append(re_item)
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


def select_evidence(
    ranked: list[RankedEvidence],
    *,
    top_k: int = TOP_K,
    max_per_family: int = MAX_PER_FAMILY,
) -> list[RankedEvidence]:
    """Select decision-grade evidence: STRONG + USABLE, diverse, top-k.

    WEAK/DISCARD are never selected. Source-family diversity caps how many hits
    from one family (e.g. Wikipedia) can be selected before an independent
    source is preferred. Mutates ``rank``/``discarded`` on the input items for
    observability and returns the selected subset in rank order.
    """
    selected: list[RankedEvidence] = []
    family_counts: dict[str, int] = {}
    deferred: list[RankedEvidence] = []

    for item in ranked:
        if item.discarded:
            continue
        if item.tier in (WEAK, DISCARD):
            item.discarded = True
            item.discard_reason = item.discard_reason or f"tier_{item.tier.lower()}"
            continue
        fam = item.family or "unknown"
        if family_counts.get(fam, 0) >= max_per_family:
            item.discard_reason = "source_family_cap"
            deferred.append(item)
            continue
        selected.append(item)
        family_counts[fam] = family_counts.get(fam, 0) + 1
        if len(selected) >= top_k:
            break

    # Backfill from family-capped items only if we are below the minimum useful
    # count and have nothing else — diversity is preferred but not at the cost of
    # having too little evidence to decide.
    if len(selected) < MIN_K:
        for item in deferred:
            if len(selected) >= MIN_K:
                break
            item.discard_reason = ""
            selected.append(item)

    for rank_i, item in enumerate(selected, start=1):
        item.rank = rank_i
        item.discarded = False
    # Mark everything not selected as discarded for the audit trail.
    sel_ids = {id(s) for s in selected}
    for item in ranked:
        if id(item) not in sel_ids and not item.discarded:
            item.discarded = True
            item.discard_reason = item.discard_reason or "not_selected"
    return selected


def rank_and_select(
    passages: Iterable[Any],
    *,
    claim_text: str,
    search_queries: Iterable[str] = (),
    domain: str = "general",
    top_k: int = TOP_K,
) -> tuple[list[RankedEvidence], list[RankedEvidence]]:
    """Convenience: rank then select. Returns (selected, all_ranked)."""
    ranked = rank_evidence(
        passages, claim_text=claim_text, search_queries=search_queries, domain=domain
    )
    selected = select_evidence(ranked, top_k=top_k)
    return selected, ranked


__all__ = [
    "STRONG",
    "USABLE",
    "WEAK",
    "DISCARD",
    "WEIGHTS",
    "AUTHORITY_PRIORS",
    "RankedEvidence",
    "score_passage",
    "rank_evidence",
    "select_evidence",
    "rank_and_select",
    "source_family",
]
