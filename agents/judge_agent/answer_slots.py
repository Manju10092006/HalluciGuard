"""Conservative answer-slot matching for the opt-in release policy.

Matching identifies a need for evidence, never a factual contradiction. Distinct
cofounders remain possible. Both must be verified when both are released.
"""
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class AnswerSlot:
    subject: str
    predicate: str
    value: str
    temporal_scope: str
    negated: bool
    hedged: bool

def normalize_entity(text, aliases=None):
    text=' '.join(re.findall(r'[a-z0-9]+',text.lower()))
    text=re.sub(r'^(?:the|a|an)\s+','',text)
    text=re.sub(r'\s+(?:inc|incorporated|corporation|corp|ltd|limited)$','',text)
    aliases=aliases or {}
    return aliases.get(text,text)

def extract_answer_slots(text, aliases=None):
    clean=text.strip().strip('.!?')
    years=re.findall(r'\b(?:1[0-9]{3}|20[0-9]{2})\b',clean)
    scope=','.join(years) if years else ('past' if re.search(r'\b(was|were|formerly|previously)\b',clean,re.I) else 'current')
    neg=bool(re.search(r"\b(?:not|never|no longer)\b|n't\b",clean,re.I))
    hedge=bool(re.search(r'\b(?:may|might|could|perhaps|possibly|reportedly)\b',clean,re.I))
    normalized=re.sub(r"\b(?:does|did|do)\s+not\s+",'',clean,flags=re.I)
    normalized=re.sub(r"\b(?:not|never|currently|formerly|reportedly|possibly)\s+",'',normalized,flags=re.I)
    normalized=re.sub(r'\b(?:may|might|could)\s+be\b','is',normalized,flags=re.I)
    normalized=re.sub(r'\b(?:may|might|could)\s+','',normalized,flags=re.I)
    patterns=[
        (r'(.+?)\s+(?:is|was|serves as)\s+(?:the\s+)?(?:current\s+)?capital\s+of\s+(.+)', 'capital', True),
        (r'(?:the\s+)?capital\s+of\s+(.+?)\s+(?:is|was)\s+(.+)', 'capital', False),
        (r'(.+?)\s+(?:was|is|were)\s+(?:co[ -]?)?(?:founded|created|developed|invented|established|designed)\s+by\s+(.+)', 'creator', False),
        (r'(.+?)\s+(?:co[ -]?)?(?:founded|created|developed|invented|established|designed|found|create|develop|invent|establish|design)\s+(.+)', 'creator', True),
        (r'(.+?)\s+(?:is|was)\s+(?:a|the|one of the)?\s*(?:co[ -]?)?(?:founders?|creators?|inventors?)\s+of\s+(.+)', 'creator', True),
        (r'(.+?)\s+(?:is|was)\s+(?:the\s+)?lead\s+developer\s+of\s+(.+)', 'lead_developer', True),
        (r"(.+?)\s+(?:is|was)\s+(.+?)['’]s\s+lead\s+developer", 'lead_developer', True),
        (r"(.+?)['’]s\s+lead\s+developer\s+(?:is|was)\s+(.+)", 'lead_developer', False),
        (r'(.+?)\s+(?:is|was)\s+(?:based|located|situated|headquartered)\s+in\s+(.+)', 'location', False),
        (r'(.+?)\s+(?:lies|lie)\s+in\s+(.+)', 'location', False),
    ]
    slots=[]
    for pattern,predicate,reverse in patterns:
        match=re.fullmatch(pattern,normalized,flags=re.I)
        if match:
            subject,value=(match[2],match[1]) if reverse else (match[1],match[2])
            subject=re.split(r'\s+(?:in|during|as of)\s+\d{4}',subject,flags=re.I)[0]
            value=re.split(r'\s+(?:in|during|as of)\s+\d{4}',value,flags=re.I)[0]
            # A founding event is not an office that changes with present tense.
            temporal='event' if predicate=='creator' and not years else scope
            slots.append(AnswerSlot(normalize_entity(subject,aliases),predicate,
                                    normalize_entity(value,aliases),temporal,neg,hedge))
    return slots

def same_answer_slot(unresolved, verified_texts, aliases=None):
    left=extract_answer_slots(unresolved,aliases)
    right=[s for text in verified_texts for s in extract_answer_slots(text,aliases)]
    return any(a.subject==b.subject and a.predicate==b.predicate
               and a.temporal_scope==b.temporal_scope for a in left for b in right)
