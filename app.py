"""
app.py — Streamlit Frontend for Multimodal RAG with Groq
Features: PDF Uploader, Extracted Assets Gallery, Interactive Chat, and Visual Citations.
"""
import os
import shutil
from pathlib import Path
import streamlit as st
from PIL import Image

from src.config import UPLOADS_DIR, GROQ_API_KEY
from src.pipeline import MultimodalRAGPipeline

# Page setup
st.set_page_config(
    page_title="Multimodal RAG with Groq",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .citation-badge {
        display: inline-block;
        padding: 4px 10px;
        margin: 2px 4px;
        background-color: #EEF2FF;
        color: #4338CA;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #C7D2FE;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize Session State
if "pipeline" not in st.session_state:
    st.session_state.pipeline = MultimodalRAGPipeline(groq_api_key=GROQ_API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = []

pipeline: MultimodalRAGPipeline = st.session_state.pipeline

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key_input = st.text_input(
        "Groq API Key",
        value=pipeline.api_key or "",
        type="password",
        placeholder="gsk_...",
        help="Get your free key from https://console.groq.com",
    )
    if api_key_input != pipeline.api_key:
        pipeline.set_api_key(api_key_input)
        st.success("API Key updated!")

    st.markdown("---")
    st.header("📁 Document Ingestion")

    uploaded_file = st.file_uploader(
        "Upload Technical / Financial PDF",
        type=["pdf"],
        help="Supports PDFs with text, tables, graphs, and images",
    )

    if uploaded_file is not None:
        if st.button("🚀 Process & Index Multimodal PDF", use_container_width=True):
            save_path = UPLOADS_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_ui(msg, pct):
                status_text.text(msg)
                progress_bar.progress(pct)

            with st.spinner("Extracting text, tables, and visual charts..."):
                result = pipeline.ingest_pdf(str(save_path), progress_callback=update_ui)

            if result["status"] == "success":
                st.success(f"Indexed **{result['document']}** successfully!")
                st.json(result["breakdown"])
            else:
                st.warning(result.get("message", "Ingestion completed with warnings."))

    st.markdown("---")
    # Document Catalog Stats
    stats = pipeline.store.get_stats()
    st.subheader("📊 Indexed Knowledge Base")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div class='metric-card'><b>Documents</b><br><h3>{len(stats['documents'])}</h3></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='metric-card'><b>Total Elements</b><br><h3>{stats['total_elements']}</h3></div>",
            unsafe_allow_html=True,
        )

    if stats["documents"]:
        st.write("Indexed files:")
        for doc in stats["documents"]:
            st.caption(f"• {doc}")

    # Clear button
    if stats["total_elements"] > 0:
        if st.button("🗑️ Clear Indexed Store", use_container_width=True):
            pipeline.store.clear()
            st.session_state.messages = []
            st.rerun()

    # Asset Gallery (Images & Tables)
    if stats["total_elements"] > 0:
        st.markdown("---")
        st.subheader("🖼️ Extracted Assets Gallery")
        gallery_tab1, gallery_tab2 = st.tabs(["Charts & Figures", "Tables"])

        with gallery_tab1:
            images = [
                v for v in pipeline.store.raw_docstore.values()
                if v.get("element_type") in ["image", "scanned_page"] and v.get("image_path")
            ]
            if images:
                for img_data in images[:6]:
                    p = img_data.get("image_path")
                    if p and os.path.exists(p):
                        st.caption(f"Page {img_data.get('page_number')} — {Path(p).name}")
                        st.image(p, use_container_width=True)
                        ocr_txt = img_data.get("metadata", {}).get("ocr_text")
                        if ocr_txt and ocr_txt.strip():
                            with st.expander("🔤 Tesseract OCR Text"):
                                st.code(ocr_txt)
            else:
                st.caption("No embedded images detected.")

        with gallery_tab2:
            tables = [
                v for v in pipeline.store.raw_docstore.values()
                if v.get("element_type") == "table"
            ]
            if tables:
                for tbl_data in tables[:4]:
                    st.caption(f"Page {tbl_data.get('page_number')} Table")
                    raw_tbl = tbl_data.get("raw_content", "")
                    if raw_tbl:
                        st.markdown(raw_tbl)
                    p = tbl_data.get("image_path")
                    if p and os.path.exists(p):
                        st.image(p, use_container_width=True)
            else:
                st.caption("No tables detected.")


# ── Main Content Area ────────────────────────────────────────────────────────
st.markdown("<div class='main-header'>📑 Multimodal PDF RAG Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Chat with your PDFs: Answers grounded in <b>Text, Tables, Charts, Figures, and Scanned Pages</b> powered by Groq Qwen.</div>",
    unsafe_allow_html=True,
)

# Suggested prompt pills
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    if st.button("📊 Explain the charts and graphs", use_container_width=True):
        st.session_state.preset_query = "What do the graphs, charts, or visual figures in this document represent?"
with col_p2:
    if st.button("📋 Compare table values", use_container_width=True):
        st.session_state.preset_query = "Summarize and compare the key metrics found in the tables of this document."
with col_p3:
    if st.button("🎯 State the core conclusion", use_container_width=True):
        st.session_state.preset_query = "What is the primary conclusion and key findings of this document?"

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            badges = "".join(
                [
                    f"<span class='citation-badge'>📄 Page {c.get('page')} ({c.get('type')})</span>"
                    for c in msg["citations"]
                ]
            )
            st.markdown(f"<div style='margin-top: 8px;'><b>Sources:</b> {badges}</div>", unsafe_allow_html=True)

        # Show visual assets attached to answer
        if "visual_assets" in msg and msg["visual_assets"]:
            with st.expander("🔍 View Referenced Visual Context (Images/Tables)", expanded=True):
                v_cols = st.columns(len(msg["visual_assets"]))
                for idx, asset_path in enumerate(msg["visual_assets"]):
                    if os.path.exists(asset_path):
                        with v_cols[idx]:
                            st.image(asset_path, caption=Path(asset_path).name, use_container_width=True)

# User Chat Input
preset = st.session_state.pop("preset_query", None)
user_query = st.chat_input("Ask a question about the document text, tables, or charts...") or preset

if user_query:
    if stats["total_elements"] == 0:
        st.warning("Please upload and index a PDF document from the sidebar first!")
    else:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document context and visuals via Groq..."):
                response = pipeline.query(user_query)
                answer = response["answer"]
                citations = response["citations"]

                # Collect visual assets to display
                visual_paths = []
                for elem in response.get("retrieved_elements", []):
                    img_p = elem.get("image_path")
                    if img_p and os.path.exists(img_p) and img_p not in visual_paths:
                        visual_paths.append(img_p)

                st.markdown(answer)

                if citations:
                    badges = "".join(
                        [
                            f"<span class='citation-badge'>📄 Page {c.get('page')} ({c.get('type')})</span>"
                            for c in citations
                        ]
                    )
                    st.markdown(f"<div style='margin-top: 8px;'><b>Sources:</b> {badges}</div>", unsafe_allow_html=True)

                if visual_paths:
                    with st.expander("🔍 View Referenced Visual Context (Images/Tables)", expanded=True):
                        v_cols = st.columns(len(visual_paths))
                        for idx, asset_path in enumerate(visual_paths):
                            with v_cols[idx]:
                                st.image(asset_path, caption=Path(asset_path).name, use_container_width=True)

        # Append Assistant Message to History
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "visual_assets": visual_paths,
            }
        )
