"""
Configuration management for Multimodal RAG with Groq
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", BASE_DIR / "data"))

# Subdirectories for multimodal artifacts
UPLOADS_DIR = STORAGE_DIR / "uploads"
IMAGES_DIR = STORAGE_DIR / "extracted_images"
TABLES_DIR = STORAGE_DIR / "extracted_tables"
STORE_DIR = STORAGE_DIR / "vector_store"

# Ensure directories exist
for path in [STORAGE_DIR, UPLOADS_DIR, IMAGES_DIR, TABLES_DIR, STORE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Groq API configuration (strip any stray quotes or spaces)
raw_key = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEY = raw_key.strip().strip('"').strip("'").strip()

# Active multimodal vision model on Groq (Llama-3.2 Vision was decommissioned by Groq)
raw_vision_model = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
if "llama-3.2" in raw_vision_model:
    GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
else:
    GROQ_VISION_MODEL = raw_vision_model

raw_chat_model = os.getenv("GROQ_CHAT_MODEL", "qwen/qwen3.8-27b")
if "llama-3.2" in raw_chat_model:
    GROQ_CHAT_MODEL = "qwen/qwen3.8-27b"
else:
    GROQ_CHAT_MODEL = raw_chat_model

# Default Chunking & Retrieval parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "4"))
