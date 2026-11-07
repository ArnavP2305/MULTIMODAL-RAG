"""
retriever.py — Hybrid Multimodal Retriever & Reranker
Uses SQLite FTS5 for native BM25 lexical search and vectorized cosine similarity,
fused with Reciprocal Rank Fusion (RRF) and page-level targeting.
"""
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict
from .store import MultimodalStore


class HybridMultimodalRetriever:
    """
    Performs true hybrid search:
    1. Dense Vector Cosine Similarity
    2. SQLite FTS5 BM25 Keyword Search
    3. Reciprocal Rank Fusion (RRF)
    4. Explicit page-target boosting
    """

    def __init__(self, store: MultimodalStore, default_top_k: int = 4):
        self.store = store
        self.default_top_k = default_top_k

    def _detect_page_target(self, query: str) -> Optional[int]:
        """Detect if the user query explicitly targets a specific page number."""
        match = re.search(r"\bpage\s*(\d+)\b", query, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Execute hybrid search and return rich multimodal element dictionaries.
        """
        k = top_k or self.default_top_k

        # 1. Vector Search (returns [(element_id, score), ...])
        vector_results = self.store.vector_search(query, top_k=k * 2)
        vector_ranked_ids = [r[0] for r in vector_results]

        # 2. Native SQLite FTS5 BM25 Search
        bm25_results = self.store.bm25_search(query, top_k=k * 2)
        bm25_ranked_ids = [r[0] for r in bm25_results]

        # If both are empty, return nothing
        if not vector_ranked_ids and not bm25_ranked_ids:
            return []

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = defaultdict(float)
        k_const = 60

        for rank, doc_id in enumerate(vector_ranked_ids):
            rrf_scores[doc_id] += 1.0 / (k_const + rank + 1)

        for rank, doc_id in enumerate(bm25_ranked_ids):
            rrf_scores[doc_id] += 1.0 / (k_const + rank + 1)

        # Check if query asks for a specific page (e.g. "Explain graph on page 2")
        target_page = self._detect_page_target(query)

        # Sort by RRF score with priority boost for page matches
        final_ranked = []
        for doc_id, score in rrf_scores.items():
            element = self.store.get_element(doc_id)
            if not element:
                continue

            final_score = score
            if target_page and element.get("page_number") == target_page:
                final_score += 1.5  # Strong boost for explicitly requested page

            final_ranked.append((element, final_score))

        final_ranked.sort(key=lambda x: x[1], reverse=True)

        return [item[0] for item in final_ranked[:k]]
