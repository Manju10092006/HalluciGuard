import math
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from ..utils import init_dll_paths
init_dll_paths()

import torch
import torch.nn.functional as F
from ..config import DetectorConfig
from ..model_manager import ModelManager


class SelfConsistencyMetrics(BaseModel):
    """Structured output containing self-consistency metrics across sampled candidate responses."""

    pairwise_similarity: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Average pairwise cosine similarity across all sampled candidate response pairs."
    )
    consistency_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized consensus score bounded in [0.0, 1.0] where 1.0 indicates complete agreement."
    )
    response_variance: float = Field(
        ...,
        ge=0.0,
        description="Semantic dispersion / variance across candidate responses (1.0 - pairwise_similarity)."
    )


class SelfConsistencyCalculator:
    """Computes multi-response self-consistency agreement across sampled candidate LLM outputs."""

    def __init__(self, model_manager: ModelManager, config: Optional[DetectorConfig] = None) -> None:
        """Initialize SelfConsistencyCalculator with shared ModelManager and optional DetectorConfig."""
        self.model_manager: ModelManager = model_manager
        self.config: DetectorConfig = config or DetectorConfig()
        self._fallback_tokenizer = None
        self._fallback_model = None

    def _get_fallback_embedding_model(self):
        """Native Hugging Face transformers fallback when sentence_transformers is not installed."""
        if self._fallback_tokenizer is None or self._fallback_model is None:
            from transformers import AutoModel, AutoTokenizer
            model_name = self.model_manager.embedding_model_name
            if not model_name.startswith("sentence-transformers/"):
                model_name = f"sentence-transformers/{model_name}"
            self._fallback_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._fallback_model = AutoModel.from_pretrained(model_name)
            self._fallback_model.eval()
        return self._fallback_tokenizer, self._fallback_model

    def _encode_responses(self, responses: List[str]) -> torch.Tensor:
        """Helper to encode a list of text responses into a normalized embedding tensor [N, D]."""
        embedding_model = self.model_manager.get_embedding_model()
        if embedding_model is not None:
            embeddings = embedding_model.encode(responses, convert_to_tensor=True)
            if len(embeddings.shape) == 1:
                embeddings = embeddings.unsqueeze(0)
            return F.normalize(embeddings, p=2, dim=1)

        tokenizer, model = self._get_fallback_embedding_model()
        inputs = tokenizer(responses, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            token_embeddings = outputs[0]
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            return F.normalize(embeddings, p=2, dim=1)

    def generate_candidate_responses(
        self,
        user_query: str,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None
    ) -> List[str]:
        """Generate N candidate responses using stochastic sampling parameters from DetectorConfig."""
        num_samples = num_samples if num_samples is not None else self.config.num_samples
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens

        tokenizer, model = self.model_manager.get_model_and_tokenizer()
        if tokenizer is None or model is None:
            return []

        prompt_text = f"User: {user_query}\nAssistant:"
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_samples,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id
            )

        responses = []
        for output in outputs:
            gen_tokens = output[input_len:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            responses.append(text if text else "N/A")

        return responses

    def compute_from_responses(self, responses: List[str]) -> Tuple[Optional[SelfConsistencyMetrics], List[List[float]]]:
        """Compute self-consistency metrics and pairwise similarity matrix from a list of responses."""
        if not responses or len(responses) < 2:
            return None, []

        try:
            embeddings = self._encode_responses(responses)
            num_resps = len(responses)

            similarity_matrix = [[0.0] * num_resps for _ in range(num_resps)]
            pairwise_sims = []

            for i in range(num_resps):
                similarity_matrix[i][i] = 1.0
                for j in range(i + 1, num_resps):
                    sim = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
                    sim_rounded = round(sim, 6)
                    similarity_matrix[i][j] = sim_rounded
                    similarity_matrix[j][i] = sim_rounded
                    pairwise_sims.append(sim)

            avg_pairwise_sim = sum(pairwise_sims) / len(pairwise_sims)
            consistency_score = (avg_pairwise_sim + 1.0) / 2.0
            consistency_score = max(0.0, min(1.0, consistency_score))
            response_variance = max(0.0, 1.0 - avg_pairwise_sim)

            metrics = SelfConsistencyMetrics(
                pairwise_similarity=round(avg_pairwise_sim, 6),
                consistency_score=round(consistency_score, 6),
                response_variance=round(response_variance, 6)
            )

            return metrics, similarity_matrix

        except Exception as e:
            print(f"[SelfConsistencyCalculator] Warning: Error computing metrics: {e}")
            return None, []

    def compute(
        self,
        user_query: str,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None
    ) -> Tuple[Optional[SelfConsistencyMetrics], List[str], List[List[float]]]:
        """Sample N responses and compute self-consistency metrics and similarity matrix."""
        responses = self.generate_candidate_responses(
            user_query=user_query,
            num_samples=num_samples,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens
        )
        metrics, matrix = self.compute_from_responses(responses)
        return metrics, responses, matrix
