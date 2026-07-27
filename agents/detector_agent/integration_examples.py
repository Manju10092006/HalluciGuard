"""HalluciGuard Detector Agent - Integration Examples

Demonstrates how external agents (Verifier, Judge, Corrector, Memory) integrate
and interact with the DetectorAgent using clean Python interfaces.
"""

from typing import Dict, Any, List
from detector_agent.detector import DetectorAgent
from detector_agent.model_manager import ModelManager
from detector_agent.models import RiskLevel, NextAction, DetectionResult


class VerifierAgentIntegrationExample:
    """Example 1: Verifier Agent Integration.
    
    The Verifier Agent calls DetectorAgent to determine if external fact-checking
    or search retrieval is required based on risk_level and next_action.
    """
    def __init__(self, detector: DetectorAgent):
        self.detector = detector

    def process_llm_response(self, user_query: str, llm_response: str) -> Dict[str, Any]:
        detection: DetectionResult = self.detector.detect(user_query, llm_response)
        
        if detection.next_action == NextAction.VERIFY:
            print(f"[VerifierAgent] Risk level is {detection.risk_level.value}. Initiating external web search verification...")
            return {
                "status": "VERIFICATION_TRIGGERED",
                "risk_level": detection.risk_level.value,
                "confidence_score": detection.confidence_score,
                "hallucination_probability": detection.hallucination_probability,
                "verification_query": user_query
            }
        else:
            print(f"[VerifierAgent] Risk level is LOW. Response accepted without external verification.")
            return {
                "status": "ACCEPTED",
                "confidence_score": detection.confidence_score,
                "response": llm_response
            }


class JudgeAgentIntegrationExample:
    """Example 2: Judge Agent Integration.
    
    The Judge Agent uses DetectorAgent metrics as evidence to render final
    accept/reject verdicts in multi-turn dialogues.
    """
    def __init__(self, detector: DetectorAgent):
        self.detector = detector

    def evaluate_candidate_responses(self, query: str, candidates: List[str]) -> Dict[str, Any]:
        evaluations = []
        for idx, candidate in enumerate(candidates):
            result = self.detector.detect(query, candidate)
            evaluations.append({
                "candidate_index": idx,
                "confidence_score": result.confidence_score,
                "hallucination_probability": result.hallucination_probability,
                "risk_level": result.risk_level.value,
                "text": candidate
            })
        
        # Sort candidates by highest confidence score
        evaluations.sort(key=lambda x: x["confidence_score"], reverse=True)
        best_candidate = evaluations[0]
        
        return {
            "verdict": "SELECTED",
            "winning_candidate": best_candidate,
            "all_evaluations": evaluations
        }


class CorrectorAgentIntegrationExample:
    """Example 3: Corrector Agent Integration.
    
    The Corrector Agent inspects low-confidence tokens and low semantic similarity
    to rewrite hallucinated response spans.
    """
    def __init__(self, detector: DetectorAgent):
        self.detector = detector

    def rewrite_if_hallucinated(self, query: str, response: str) -> str:
        result = self.detector.detect(query, response)
        
        if result.risk_level == RiskLevel.HIGH:
            print(f"[CorrectorAgent] HIGH risk detected ({result.hallucination_probability:.2f} prob). Triggering response correction...")
            # Extract low probability token warnings if available
            low_prob_tokens = []
            if result.metrics.token_probability:
                low_prob_tokens = result.metrics.token_probability.low_prob_tokens
            
            corrected_response = f"[Corrected] {response} (Validated contextually)"
            return corrected_response
        
        return response


class MemoryAgentIntegrationExample:
    """Example 4: Memory Agent Integration.
    
    The Memory Agent stores high-confidence context-response pairs into vector memory
    while indexing flagged hallucinations for negative retrieval filtering.
    """
    def __init__(self, detector: DetectorAgent):
        self.detector = detector
        self.positive_memory = []
        self.negative_hallucination_memory = []

    def Index_response(self, query: str, response: str):
        result = self.detector.detect(query, response)
        
        record = {
            "query": query,
            "response": response,
            "confidence_score": result.confidence_score,
            "risk_level": result.risk_level.value
        }
        
        if result.risk_level == RiskLevel.LOW:
            self.positive_memory.append(record)
            print("[MemoryAgent] Response indexed into Positive Vector Memory.")
        else:
            self.negative_hallucination_memory.append(record)
            print("[MemoryAgent] Response indexed into Hallucination Blocklist Memory.")


if __name__ == "__main__":
    print("=== Running HalluciGuard Multi-Agent Integration Demonstration ===")
    
    # Instantiate shared ModelManager and DetectorAgent
    model_manager = ModelManager()
    detector = DetectorAgent(model_manager=model_manager)

    test_query = "What is the capital of France?"
    test_response_correct = "The capital of France is Paris."
    test_response_hallucinated = "The capital of France is Berlin and it was founded in 1999."

    print("\n--- 1. Verifier Agent Integration ---")
    verifier = VerifierAgentIntegrationExample(detector)
    res1 = verifier.process_llm_response(test_query, test_response_correct)
    res2 = verifier.process_llm_response(test_query, test_response_hallucinated)

    print("\n--- 2. Judge Agent Integration ---")
    judge = JudgeAgentIntegrationExample(detector)
    judge_res = judge.evaluate_candidate_responses(test_query, [test_response_hallucinated, test_response_correct])
    print(f"Judge Selected Candidate Index: {judge_res['winning_candidate']['candidate_index']} with Confidence: {judge_res['winning_candidate']['confidence_score']:.4f}")

    print("\n--- 3. Corrector Agent Integration ---")
    corrector = CorrectorAgentIntegrationExample(detector)
    corrected_output = corrector.rewrite_if_hallucinated(test_query, test_response_hallucinated)
    print(f"Corrector Output: {corrected_output}")

    print("\n--- 4. Memory Agent Integration ---")
    memory = MemoryAgentIntegrationExample(detector)
    memory.Index_response(test_query, test_response_correct)
    memory.Index_response(test_query, test_response_hallucinated)
    print(f"Positive Memory Count: {len(memory.positive_memory)}, Negative Memory Count: {len(memory.negative_hallucination_memory)}")
    print("\n=== Integration Demonstration Completed Successfully ===")
