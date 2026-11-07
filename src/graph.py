"""
graph.py — LangGraph Multimodal RAG State Machine
Orchestrates Query Routing, Hybrid Retrieval, Image Payload Assembly,
and Vision-Language Synthesis via Groq.
"""
import os
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from groq import Groq

from .config import GROQ_API_KEY, GROQ_VISION_MODEL, GROQ_CHAT_MODEL
from .retriever import HybridMultimodalRetriever
from .store import MultimodalStore


class MultimodalRAGState(TypedDict):
    """Workflow state passing through LangGraph nodes."""
    question: str
    chat_history: List[Dict[str, str]]
    route: str  # "visual" | "table" | "general"
    retrieved_elements: List[Dict[str, Any]]
    image_payloads: List[str]  # base64 strings of retrieved images/charts
    citations: List[Dict[str, Any]]
    answer: str


def build_multimodal_graph(store: MultimodalStore, groq_api_key: Optional[str] = None):
    """Factory creating the LangGraph compiled state machine."""
    api_key = groq_api_key or GROQ_API_KEY
    retriever = HybridMultimodalRetriever(store)
    client = Groq(api_key=api_key) if api_key else None

    # ── Node 1: Router Node ──────────────────────────────────────────────────
    def router_node(state: MultimodalRAGState) -> Dict[str, Any]:
        """Classify user intent: asking for visual figures, tables, or general text."""
        q = state["question"].lower()
        if any(w in q for w in ["graph", "chart", "figure", "image", "diagram", "plot", "picture", "architecture"]):
            route = "visual"
        elif any(w in q for w in ["table", "row", "column", "data", "metric", "percentage", "compare values"]):
            route = "table"
        else:
            route = "general"
        return {"route": route}

    # ── Node 2: Retrieval Node ───────────────────────────────────────────────
    def retrieve_node(state: MultimodalRAGState) -> Dict[str, Any]:
        """Retrieve relevant text chunks, markdown tables, and charts."""
        elements = retriever.retrieve(state["question"], top_k=4)

        image_payloads = []
        citations = []

        for elem in elements:
            # Record citation metadata
            citations.append({
                "page": elem.get("page_number", 1),
                "type": elem.get("element_type", "text"),
                "source": elem.get("source_doc", "document"),
                "id": elem.get("element_id"),
                "image_path": elem.get("image_path")
            })

            # If element has an associated image or table snapshot, encode it for Groq Vision
            img_path = elem.get("image_path")
            if img_path and os.path.exists(img_path) and len(image_payloads) < 2:
                try:
                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                        image_payloads.append(b64)
                except Exception:
                    pass

        return {
            "retrieved_elements": elements,
            "image_payloads": image_payloads,
            "citations": citations,
        }

    # ── Node 3: Synthesis Node (Groq Vision + LangChain) ──────────────────────
    def synthesize_node(state: MultimodalRAGState) -> Dict[str, Any]:
        """Generate final answer using Groq Llama 3.2 Vision."""
        if not client:
            return {
                "answer": (
                    "⚠️ **Groq API key not configured.**\n\n"
                    "Please set `GROQ_API_KEY` in your `.env` file or provide it in the web interface to enable multimodal answers."
                )
            }

        elements = state.get("retrieved_elements", [])
        if not elements:
            return {"answer": "I could not find any relevant text, tables, or figures in the uploaded documents."}

        # Format context from retrieved multimodal elements
        context_blocks = []
        for i, elem in enumerate(elements, 1):
            e_type = elem.get("element_type", "text")
            page = elem.get("page_number", "?")
            doc = elem.get("source_doc", "doc")

            if e_type == "table":
                raw_table = elem.get("raw_content", "")
                summary = elem.get("searchable_summary", "")
                context_blocks.append(
                    f"[Source #{i} - Table on Page {page} ({doc})]\n"
                    f"Summary: {summary}\n"
                    f"Structured Data:\n{raw_table}"
                )
            elif e_type in ["image", "scanned_page"]:
                caption = elem.get("searchable_summary", "")
                context_blocks.append(
                    f"[Source #{i} - {e_type.title()} on Page {page} ({doc})]\n"
                    f"Visual Analysis: {caption}"
                )
            else:
                text = elem.get("raw_content", "")
                context_blocks.append(
                    f"[Source #{i} - Text excerpt on Page {page} ({doc})]\n{text}"
                )

        full_context = "\n\n---\n\n".join(context_blocks)

        system_prompt = (
            "You are an elite Multimodal Document Research Assistant.\n"
            "You answer questions with rigorous precision using the provided text, structured tables, and visual charts.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Ground your answer strictly in the provided context and images.\n"
            "2. When citing facts, tables, or graphs, always cite the page number explicitly (e.g., [Page 4, Table 1]).\n"
            "3. If comparing values across tables or charts, quote the exact numbers and units.\n"
            "4. If the information is not present, state clearly that it is not found in the documents."
        )

        user_content = [
            {
                "type": "text",
                "text": f"CONTEXT FROM UPLOADED DOCUMENTS:\n{full_context}\n\nQUESTION:\n{state['question']}"
            }
        ]

        # Attach image payloads for Groq Vision model
        for b64 in state.get("image_payloads", []):
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

        try:
            chosen_model = GROQ_VISION_MODEL if state.get("image_payloads") else GROQ_CHAT_MODEL
            response = client.chat.completions.create(
                model=chosen_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=1000,
                temperature=0.1,
            )
            answer_text = response.choices[0].message.content.strip()
        except Exception as e:
            err_msg = str(e)
            # If decommissioned model was requested, automatically fallback to qwen/qwen3.8-27b
            if "decommissioned" in err_msg or "llama-3.2" in err_msg:
                try:
                    response = client.chat.completions.create(
                        model="qwen/qwen3.8-27b",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        max_tokens=1000,
                        temperature=0.1,
                    )
                    answer_text = response.choices[0].message.content.strip()
                except Exception as fallback_err:
                    answer_text = f"Error during Groq generation: {str(fallback_err)}"
            else:
                answer_text = f"Error during Groq generation: {err_msg}"

        return {"answer": answer_text}

    # ── Compile Workflow ──────────────────────────────────────────────────────
    workflow = StateGraph(MultimodalRAGState)
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("synthesizer", synthesize_node)

    workflow.add_edge(START, "router")
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()
