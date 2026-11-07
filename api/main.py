"""
main.py — FastAPI Backend for Multimodal RAG
Provides endpoints for PDF uploading, indexing, and chat queries.
"""
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import UPLOADS_DIR, IMAGES_DIR, TABLES_DIR
from src.pipeline import MultimodalRAGPipeline

app = FastAPI(
    title="Multimodal RAG API",
    description="Multimodal PDF RAG with Text, Tables, Charts, and Groq Vision LLMs",
    version="1.0.0",
)

# Enable CORS for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount extracted images and tables as static assets
app.mount("/static/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
app.mount("/static/tables", StaticFiles(directory=str(TABLES_DIR)), name="tables")

# Pipeline singleton
pipeline = MultimodalRAGPipeline()


# ── Pydantic Request / Response Schemas ──────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]] = []
    groq_api_key: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    route: str
    citations: List[Dict[str, Any]]
    retrieved_count: int


# ── API Endpoints ────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    stats = pipeline.store.get_stats()
    return {
        "status": "healthy",
        "groq_configured": bool(pipeline.api_key),
        "indexed_stats": stats,
    }


@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    groq_api_key: Optional[str] = Form(None),
):
    """Upload a PDF, extract multimodal elements, and index them."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if groq_api_key:
        pipeline.set_api_key(groq_api_key)

    dest_path = UPLOADS_DIR / file.filename
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = pipeline.ingest_pdf(str(dest_path))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Ask a question about the uploaded documents."""
    if request.groq_api_key:
        pipeline.set_api_key(request.groq_api_key)

    try:
        response = pipeline.query(request.question, request.chat_history)
        return ChatResponse(
            answer=response["answer"],
            route=response["route"],
            citations=response["citations"],
            retrieved_count=len(response.get("retrieved_elements", [])),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/api/documents")
def get_documents():
    """List all indexed documents and catalog statistics."""
    return pipeline.store.get_stats()


@app.delete("/api/clear")
def clear_all():
    """Reset all indexed documents and store."""
    pipeline.store.clear()
    return {"status": "success", "message": "Catalog and vector store cleared."}
