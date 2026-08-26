from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum

class Intent(str, Enum):
    GREETING = "greeting"
    CONVERSATIONAL = "conversational"
    FACTUAL_CLAIM = "factual_claim"
    FACTUAL_QUESTION = "factual_question"
    SUBJECTIVE_QUESTION = "subjective_question"
    MEDICAL_ADVICE = "medical_advice"
    LEGAL_ADVICE = "legal_advice"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"

@dataclass(frozen=True)
class IntentDecision:
    intent: Intent
    should_verify: bool
    response: str | None = None

_GREETINGS = {"hi", "hello", "hey", "hey there", "hi there", "good morning", "good afternoon", "good evening", "howdy", "hola", "greetings", "yo"}
_CONVERSATIONAL = {"who are you", "what are you", "what can you do", "what is halluciguard", "how does this work", "thank you", "thanks"}
_SUPERLATIVES = re.compile(r"\b(best|worst|most dangerous|greatest|largest|smallest|most effective|most popular|number one|top|strongest|weakest|fastest)\b", re.I)
_MEDICAL = re.compile(r"\b(medicine|medication|drug|dose|dosage|treatment|treat|cure|headache|symptom|diagnos(?:e|is)|prescri(?:be|ption))\b", re.I)
_LEGAL = re.compile(r"\b(legal advice|lawyer|attorney|sue|lawsuit|contract|liable|liability|court|legal|illegal)\b", re.I)
_ADVICE_FORM = re.compile(r"\b(what should i|should i|which should i|recommend|best|how do i treat|can i take)\b", re.I)
_QUESTION = re.compile(r"^(who|what|when|where|why|how|is|are|was|were|does|do|did|can|could|will|would|has|have|which)\b", re.I)

def classify_intent(text: str) -> IntentDecision:
    normalized = re.sub(r"\s+", " ", text.strip().lower().strip("!?.,'\"` "))
    if not normalized:
        return IntentDecision(Intent.UNSUPPORTED, False, "Please enter a factual claim or question to verify.")
    if normalized in _GREETINGS:
        return IntentDecision(Intent.GREETING, False, "Hello! Send me a factual claim or question and I’ll verify it against evidence.")
    if normalized in _CONVERSATIONAL:
        return IntentDecision(Intent.CONVERSATIONAL, False, "I’m HalluciGuard, an evidence-grounded factual verification system.")
    is_question = text.strip().endswith("?") or bool(_QUESTION.match(normalized))
    if is_question and _MEDICAL.search(normalized) and _ADVICE_FORM.search(normalized):
        return IntentDecision(Intent.MEDICAL_ADVICE, False, "This is a medical-guidance question, not a factual claim with a single verifiable verdict. Treatment depends on the headache type, symptoms, health conditions, and other medicines. I can verify a specific medical claim, but I can’t prescribe a medicine or dosage; consult a qualified clinician or pharmacist for individual advice.")
    if is_question and _LEGAL.search(normalized) and _ADVICE_FORM.search(normalized):
        return IntentDecision(Intent.LEGAL_ADVICE, False, "This asks for individualized legal guidance rather than verification of a specific factual proposition. Laws depend on jurisdiction and circumstances; consult a qualified legal professional.")
    if is_question and _SUPERLATIVES.search(normalized):
        return IntentDecision(Intent.SUBJECTIVE_QUESTION, False, "This question uses a subjective or superlative comparison without a defined metric, comparison set, authority, and timeframe. Please specify those criteria.")
    return IntentDecision(Intent.FACTUAL_QUESTION if is_question else Intent.FACTUAL_CLAIM, True)
