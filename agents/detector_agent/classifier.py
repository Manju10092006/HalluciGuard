from enum import Enum
import re


class QueryCategory(str, Enum):
    """Categories assigned to user prompts by the rule-based Query Classifier."""
    FACTUAL_QUESTION = "factual_question"
    EXPLANATION = "explanation"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    CREATIVE_WRITING = "creative_writing"
    STORYTELLING = "storytelling"
    BRAINSTORMING = "brainstorming"
    POETRY = "poetry"
    FICTIONAL_CONTENT = "fictional_content"
    ROLEPLAY = "roleplay"
    OTHER = "other"


def classify_query(user_query: str) -> QueryCategory:
    """Classifies a user query into one of eleven predefined categories using rule-based heuristics.
    
    Args:
        user_query: The raw prompt string submitted by the user.
        
    Returns:
        QueryCategory: Categorized prompt classification.
    """
    text = user_query.strip().lower()

    # Rule 1: Roleplay & Persona
    if re.search(r"\b(roleplay|act as|pretend you are|speak like|you are a|character)\b", text):
        return QueryCategory.ROLEPLAY

    # Rule 2: Poetry & Rhymes
    if re.search(r"\b(poem|limerick|haiku|rhyme|sonnet|verse|stanza)\b", text):
        return QueryCategory.POETRY

    # Rule 3: Storytelling
    if re.search(r"\b(tell me a story|once upon a time|narrative|chapter|fairytale|fable|fiction story)\b", text):
        return QueryCategory.STORYTELLING

    # Rule 4: Fictional & Imaginary Content
    if re.search(r"\b(fictional|imaginary|sci-fi|fantasy world|make up a|zalthor|mythical|mythology)\b", text):
        return QueryCategory.FICTIONAL_CONTENT

    # Rule 5: Creative Writing
    if re.search(r"\b(write a story|create a story|invent|creative writing|compose a|fiction|imagine a)\b", text):
        return QueryCategory.CREATIVE_WRITING

    # Rule 6: Brainstorming & Idea Generation
    if re.search(r"\b(brainstorm|give me ideas|suggest names|list ideas|generate options|ideas for)\b", text):
        return QueryCategory.BRAINSTORMING

    # Rule 7: Summarization
    if re.search(r"\b(summarize|tl;dr|summary|key points|synopsis|condense|digest)\b", text):
        return QueryCategory.SUMMARIZATION

    # Rule 8: Reasoning, Logic & Calculation
    if re.search(r"\b(calculate|solve|prove|compare|evaluate|analyze|logic behind|step-by-step|derive|math|formula|if .* then)\b", text):
        return QueryCategory.REASONING

    # Rule 9: Explanations
    if re.search(r"\b(explain|why does|how does|describe how|what causes|elaborate on|walk me through|mechanism of)\b", text):
        return QueryCategory.EXPLANATION

    # Rule 10: Factual Questions
    if re.search(r"\b(what is|who is|when did|where is|capital of|how many|which country|definition of|date of|name of|located in|born in)\b", text):
        return QueryCategory.FACTUAL_QUESTION

    # Fallback to OTHER
    return QueryCategory.OTHER
