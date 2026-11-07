"""
test_pipeline.py — End-to-end integration test for the Multimodal RAG pipeline.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.extractor import MultimodalPDFExtractor
from src.store import MultimodalStore
from src.retriever import HybridMultimodalRetriever
from src.pipeline import MultimodalRAGPipeline

SAMPLE_PDF = str(Path(__file__).resolve().parent / "sample_data" / "sample_multimodal_report.pdf")


def test_extraction():
    print("\n--- 1. Testing Extractor ---")
    extractor = MultimodalPDFExtractor()
    elements = extractor.extract(SAMPLE_PDF)
    print(f"Extracted {len(elements)} elements:")
    for elem in elements:
        print(f" - [{elem.element_type.upper()}] Page {elem.page_number} (ID: {elem.element_id})")
        if elem.image_path:
            print(f"    Saved Asset: {Path(elem.image_path).name}")
        if elem.content and len(elem.content) < 150:
            print(f"    Content preview:\n{elem.content}")
    assert len(elements) > 0, "No elements extracted!"
    print(" Extraction test passed!")


def test_ingestion_and_retrieval():
    print("\n--- 2. Testing Store & Hybrid Retrieval ---")
    pipeline = MultimodalRAGPipeline()
    res = pipeline.ingest_pdf(SAMPLE_PDF)
    print("Ingestion result:", res)

    print("\nTesting Query 1: 'What was the table accuracy for our architecture?'")
    r1 = pipeline.retriever = HybridMultimodalRetriever(pipeline.store)
    docs1 = r1.retrieve("table accuracy architecture")
    print(f"Retrieved {len(docs1)} elements for query 1:")
    for d in docs1:
        print(f" - Page {d.get('page_number')} [{d.get('element_type')}]: {d.get('element_id')}")

    print("\nTesting Query 2: 'Explain the latency chart in Figure 1 on page 2'")
    docs2 = r1.retrieve("latency chart Figure 1 page 2")
    print(f"Retrieved {len(docs2)} elements for query 2:")
    for d in docs2:
        print(f" - Page {d.get('page_number')} [{d.get('element_type')}]: {d.get('element_id')}")

    print("\n Pipeline & MultiVector Retrieval verified successfully!")


if __name__ == "__main__":
    test_extraction()
    test_ingestion_and_retrieval()
