from typing import Tuple
from .classifier import QueryCategory, classify_query
from .models import RiskLevel


class SelfConsistencyGate:
    """Intelligent gating mechanism for triggering Self-Consistency evaluation.
    
    Condition A: Current risk level must be MEDIUM.
    Condition B: Query category must be one of:
      - factual_question
      - explanation
      - summarization
      - reasoning
      
    If BOTH conditions are met -> Returns True (Execute Self-Consistency).
    Otherwise -> Returns False (Skip Self-Consistency to optimize compute).
    """

    ANALYTICAL_CATEGORIES = {
        QueryCategory.FACTUAL_QUESTION,
        QueryCategory.EXPLANATION,
        QueryCategory.SUMMARIZATION,
        QueryCategory.REASONING,
    }

    def should_run_self_consistency(
        self, user_query: str, current_risk_level: RiskLevel
    ) -> Tuple[bool, str, QueryCategory]:
        """Evaluates gating rules for triggering Self-Consistency.
        
        Args:
            user_query: The prompt text submitted to the LLM.
            current_risk_level: RiskLevel evaluated from single-pass signals.
            
        Returns:
            Tuple[bool, str, QueryCategory]: (should_run, log_reason, category)
        """
        category = classify_query(user_query)

        # Condition A: Risk level must be MEDIUM
        if current_risk_level != RiskLevel.MEDIUM:
            reason = f"[Gate] {current_risk_level.value} risk -> Skip Self-Consistency"
            return False, reason, category

        # Condition B: Query category must be analytical (factual, explanation, summarization, reasoning)
        if category not in self.ANALYTICAL_CATEGORIES:
            category_name = category.value.replace("_", " ").title()
            reason = f"[Gate] {category_name} -> Skip Self-Consistency"
            return False, reason, category

        category_name = category.value.replace("_", " ").title()
        reason = f"[Gate] MEDIUM Risk + {category_name} -> Execute Self-Consistency"
        return True, reason, category
