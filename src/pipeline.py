"""
pipeline.py — End-to-End Multimodal Ingestion & Query Coordinator with OCR Fusion
"""
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

from .config import GROQ_API_KEY
from .extractor import MultimodalPDFExtractor, ExtractedElement
from .summarizer import GroqMultimodalSummarizer
from .store import MultimodalStore
from .graph import build_multimodal_graph


class MultimodalRAGPipeline:
    """Coordinates document ingestion, OCR extraction, VLM enrichment, and question answering."""

    def __init__(self, groq_api_key: Optional[str] = None):
        self.api_key = groq_api_key or GROQ_API_KEY
        self.store = MultimodalStore()
        self.extractor = MultimodalPDFExtractor()
        self.summarizer = GroqMultimodalSummarizer(api_key=self.api_key)
        self.graph = build_multimodal_graph(self.store, groq_api_key=self.api_key)

    def set_api_key(self, api_key: str):
        """Update API key dynamically at runtime."""
        self.api_key = api_key
        self.summarizer = GroqMultimodalSummarizer(api_key=api_key)
        self.graph = build_multimodal_graph(self.store, groq_api_key=api_key)

    def ingest_pdf(
        self,
        pdf_path: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Extract multimodal elements, run OCR on images/tables, generate VLM summaries,
        and index them into the multi-vector store.
        """
        def update_progress(msg: str, pct: float):
            if progress_callback:
                progress_callback(msg, pct)

        update_progress(f"Extracting elements & running OCR on {Path(pdf_path).name}...", 0.1)
        elements: List[ExtractedElement] = self.extractor.extract(pdf_path)

        total = len(elements)
        if total == 0:
            return {"status": "warning", "message": "No elements found in document.", "elements_count": 0}

        counts = {"text": 0, "table": 0, "image": 0, "scanned_page": 0}

        for idx, elem in enumerate(elements):
            pct = 0.1 + (0.8 * (idx / total))
            counts[elem.element_type] = counts.get(elem.element_type, 0) + 1

            searchable_summary = elem.content
            ocr_text = elem.metadata.get("ocr_text", "")

            # Multimodal Enrichment via Groq VLM fused with Tesseract OCR text
            if elem.element_type == "table":
                update_progress(f"Analyzing Table on page {elem.page_number} with Groq...", pct)
                searchable_summary = self.summarizer.summarize_table(
                    elem.content, elem.page_number, ocr_text=ocr_text
                )
            elif elem.element_type == "image":
                update_progress(f"Analyzing Visual Figure on page {elem.page_number} (OCR + Groq Vision)...", pct)
                if elem.image_path:
                    searchable_summary = self.summarizer.summarize_image(
                        elem.image_path, ocr_text=ocr_text
                    )
            elif elem.element_type == "scanned_page":
                update_progress(f"Transcribing Scanned Page {elem.page_number} (Tesseract OCR + VLM)...", pct)
                if elem.image_path:
                    searchable_summary = self.summarizer.transcribe_scanned_page(
                        elem.image_path, ocr_text=ocr_text
                    )

            # Store in MultiVector Store
            self.store.add_element(
                element_id=elem.element_id,
                searchable_text=searchable_summary,
                element_type=elem.element_type,
                page_number=elem.page_number,
                source_doc=elem.source_doc,
                raw_content=elem.content,
                image_path=elem.image_path,
                metadata=elem.metadata,
            )

        # Persist catalog
        self.store.save()
        # Re-build graph with updated store
        self.graph = build_multimodal_graph(self.store, groq_api_key=self.api_key)

        update_progress("Ingestion complete!", 1.0)
        return {
            "status": "success",
            "document": Path(pdf_path).name,
            "total_elements": total,
            "breakdown": counts,
        }

    def query(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Run user question through LangGraph pipeline."""
        initial_state = {
            "question": question,
            "chat_history": chat_history or [],
            "route": "general",
            "retrieved_elements": [],
            "image_payloads": [],
            "citations": [],
            "answer": "",
        }

        result = self.graph.invoke(initial_state)

        return {
            "answer": result.get("answer", ""),
            "route": result.get("route", "general"),
            "citations": result.get("citations", []),
            "retrieved_elements": result.get("retrieved_elements", []),
        }
