"""
Embedding similarity evaluator — semantic cosine similarity via sentence-transformers.

Requires: pip install sentence-transformers>=2.7
Model is loaded lazily on first call and pinned to MODEL_NAME.
Embeddings are cached by text hash within a run to avoid redundant encoding.
"""

import hashlib
from typing import Any, Optional

import numpy as np

from sumi.evaluators.base import Evaluator
from sumi.models import TestCase, ValidationScenario


class EmbeddingEvaluator(Evaluator):
    """
    Score semantic similarity between response and reference_text via cosine similarity.

    Scores are only comparable when MODEL_NAME is held constant across runs.
    Changing the model resets the meaning of all historical scores.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._cache: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return "embedding_sim"

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embedding_sim evaluation. "
                    "Install it: pip install sentence-transformers>=2.7"
                )
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _encode(self, text: str) -> np.ndarray:
        key = hashlib.md5(text.encode()).hexdigest()
        if key not in self._cache:
            self._cache[key] = self._get_model().encode(text)
        return self._cache[key]

    def score(
        self,
        prompt: str,
        response: str,
        test_case: TestCase,
        scenario: ValidationScenario,
        conversation_history: Optional[list] = None,
    ) -> tuple[float, Optional[str]]:
        if not test_case.reference_text:
            return 0.5, "No reference_text provided — cannot compute embedding similarity"

        if not response.strip():
            return 0.0, "Empty response"

        response_emb = self._encode(response.strip())
        reference_emb = self._encode(test_case.reference_text.strip())

        norm = float(np.linalg.norm(response_emb) * np.linalg.norm(reference_emb))
        similarity = float(np.dot(response_emb, reference_emb)) / norm if norm > 0 else 0.0
        score = max(0.0, min(1.0, similarity))

        return round(score, 3), f"cosine_sim={similarity:.3f} (model={self._model_name})"
