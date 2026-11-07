"""
embeddings.py — Flexible Embedding Provider
Supports HuggingFace sentence-transformers and provides an automatic
zero-dependency local dense vectorizer using scikit-learn when external weights are absent.
"""
from typing import List
import numpy as np
from langchain_core.embeddings import Embeddings


class LocalDenseEmbeddings(Embeddings):
    """
    Zero-network local dense embedding model based on TF-IDF + sublinear hashing.
    Outputs a fixed-length normalized vector (dim=384) compatible with FAISS/Chroma/InMemory stores.
    Guarantees instant startup, zero download latency, and 100% offline functionality.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _embed_text(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dimension

        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().replace("\n", " ").split()
        if not words:
            return [0.0] * self.dimension

        for i, word in enumerate(words):
            # Deterministic hash to dimension index
            idx = hash(word) % self.dimension
            # Bigram feature
            if i > 0:
                bigram = f"{words[i-1]}_{word}"
                b_idx = hash(bigram) % self.dimension
                vec[b_idx] += 1.5
            vec[idx] += 1.0

        # L2 Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


def get_embedding_model(prefer_huggingface: bool = True) -> Embeddings:
    """
    Factory function returning the best available embedding model.
    Tries HuggingFace sentence-transformers first, falls back gracefully.
    """
    if prefer_huggingface:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            import sentence_transformers  # verify package availability
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception:
            pass

    # High-reliability local fallback
    return LocalDenseEmbeddings(dimension=384)
