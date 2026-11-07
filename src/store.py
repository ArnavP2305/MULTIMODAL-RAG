"""
store.py — Unified SQLite Hybrid Vector & Document Database
Combines:
1. SQLite relational storage for multimodal metadata, image paths, and page numbers.
2. SQLite FTS5 (Full-Text Search) with C-speed BM25 ranking for exact keyword search.
3. High-speed vectorized cosine similarity search over dense embeddings.
"""
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from langchain_core.documents import Document

from .config import STORE_DIR
from .embeddings import get_embedding_model


class MultimodalStore:
    """
    Unified SQLite Hybrid Store:
    Replaces brittle JSON docstores and heavy vector databases with an ACID-compliant,
    single-file database featuring built-in BM25 full-text search and vectorized cosine similarity.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_dir = STORE_DIR
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (self.db_dir / "multimodal.db")

        self.embedding_model = get_embedding_model()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with WAL mode enabled for high concurrency."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        """Initialize tables and full-text search virtual table."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS elements (
                    element_id TEXT PRIMARY KEY,
                    element_type TEXT,
                    page_number INTEGER,
                    source_doc TEXT,
                    raw_content TEXT,
                    searchable_summary TEXT,
                    image_path TEXT,
                    ocr_text TEXT,
                    metadata_json TEXT,
                    vector_blob BLOB
                )
                """
            )
            # FTS5 Virtual Table for BM25 Search
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_elements USING fts5(
                    element_id UNINDEXED,
                    content
                )
                """
            )
            conn.commit()

    def add_element(
        self,
        element_id: str,
        searchable_text: str,
        element_type: str,
        page_number: int,
        source_doc: str,
        raw_content: str = "",
        image_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Compute embedding and insert/update element in both relational and FTS5 tables."""
        meta = metadata or {}
        ocr_text = meta.get("ocr_text", "")

        # Compute dense embedding
        emb = self.embedding_model.embed_query(searchable_text)
        vec_bytes = np.array(emb, dtype=np.float32).tobytes()

        # Full content for BM25 search: summary + raw text + OCR tokens
        full_fts_content = f"{searchable_text} {raw_content} {ocr_text}"

        with self._get_connection() as conn:
            # Upsert into elements
            conn.execute(
                """
                INSERT INTO elements (
                    element_id, element_type, page_number, source_doc,
                    raw_content, searchable_summary, image_path, ocr_text,
                    metadata_json, vector_blob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(element_id) DO UPDATE SET
                    element_type=excluded.element_type,
                    page_number=excluded.page_number,
                    source_doc=excluded.source_doc,
                    raw_content=excluded.raw_content,
                    searchable_summary=excluded.searchable_summary,
                    image_path=excluded.image_path,
                    ocr_text=excluded.ocr_text,
                    metadata_json=excluded.metadata_json,
                    vector_blob=excluded.vector_blob
                """,
                (
                    element_id,
                    element_type,
                    page_number,
                    source_doc,
                    raw_content,
                    searchable_text,
                    image_path,
                    ocr_text,
                    json.dumps(meta),
                    vec_bytes,
                ),
            )

            # Update FTS5 index (delete existing and insert new)
            conn.execute("DELETE FROM fts_elements WHERE element_id = ?", (element_id,))
            conn.execute(
                "INSERT INTO fts_elements (element_id, content) VALUES (?, ?)",
                (element_id, full_fts_content),
            )
            conn.commit()

    def get_element(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full multimodal element record by element_id."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM elements WHERE element_id = ?", (element_id,)).fetchone()
            if not row:
                return None
            return self._row_to_dict(row)

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a SQLite row to a clean dictionary."""
        try:
            meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        except Exception:
            meta = {}

        return {
            "element_id": row["element_id"],
            "element_type": row["element_type"],
            "page_number": row["page_number"],
            "source_doc": row["source_doc"],
            "raw_content": row["raw_content"],
            "searchable_summary": row["searchable_summary"],
            "image_path": row["image_path"],
            "ocr_text": row["ocr_text"],
            "metadata": meta,
        }

    @property
    def raw_docstore(self) -> Dict[str, Dict[str, Any]]:
        """Compatibility property for UI catalog view."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM elements").fetchall()
            return {r["element_id"]: self._row_to_dict(r) for r in rows}

    def vector_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Perform vectorized cosine similarity search over all element embeddings."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT element_id, vector_blob FROM elements WHERE vector_blob IS NOT NULL").fetchall()
            if not rows:
                return []

            element_ids = [r["element_id"] for r in rows]
            # Stack all stored vectors into a 2D numpy matrix: (N, dim)
            matrix = np.vstack([np.frombuffer(r["vector_blob"], dtype=np.float32) for r in rows])

            # Compute query vector
            q_emb = np.array(self.embedding_model.embed_query(query), dtype=np.float32)
            q_norm = np.linalg.norm(q_emb)
            if q_norm > 0:
                q_emb = q_emb / q_norm

            # Matrix dot product
            scores = np.dot(matrix, q_emb)

            # Rank candidates
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            return [(element_ids[idx], float(scores[idx])) for idx in ranked_indices]

    def bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Execute true BM25 full-text search using SQLite FTS5."""
        clean_tokens = [w for w in query.replace('"', "").replace("'", "").split() if len(w) > 1]
        if not clean_tokens:
            return []

        # Join tokens with OR or match phrase
        match_query = " OR ".join(clean_tokens)
        with self._get_connection() as conn:
            try:
                # FTS5 bm25 ranking function: lower rank value is more relevant
                rows = conn.execute(
                    """
                    SELECT element_id, rank
                    FROM fts_elements
                    WHERE fts_elements MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (match_query, top_k),
                ).fetchall()
                # Normalize score (invert rank so higher is better)
                return [(r["element_id"], -float(r["rank"])) for r in rows]
            except Exception:
                # Fallback to simple LIKE search if special FTS syntax fails
                like_hits = []
                for token in clean_tokens[:3]:
                    res = conn.execute(
                        "SELECT element_id FROM elements WHERE searchable_summary LIKE ? OR raw_content LIKE ? LIMIT ?",
                        (f"%{token}%", f"%{token}%", top_k),
                    ).fetchall()
                    for r in res:
                        like_hits.append((r["element_id"], 1.0))
                return like_hits[:top_k]

    def list_documents(self) -> List[str]:
        """List distinct document names."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT DISTINCT source_doc FROM elements ORDER BY source_doc").fetchall()
            return [r["source_doc"] for r in rows if r["source_doc"]]

    def get_stats(self) -> Dict[str, Any]:
        """Get element counts and distribution."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM elements").fetchone()["c"]
            type_rows = conn.execute("SELECT element_type, COUNT(*) as c FROM elements GROUP BY element_type").fetchall()
            types = {r["element_type"]: r["c"] for r in type_rows}
            docs = self.list_documents()

            return {
                "total_elements": total,
                "by_type": types,
                "documents": docs,
            }

    def clear(self):
        """Wipe all elements from database."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM elements")
            conn.execute("DELETE FROM fts_elements")
            conn.commit()

    def save(self):
        """No-op: SQLite commits immediately to disk with ACID transactions."""
        pass
