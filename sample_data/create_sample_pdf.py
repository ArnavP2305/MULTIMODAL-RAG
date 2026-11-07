"""
create_sample_pdf.py — Generates a sample technical PDF containing:
1. Executive narrative text.
2. A structured benchmark table.
3. An embedded high-resolution Matplotlib chart.
"""
import io
from pathlib import Path
import matplotlib.pyplot as plt
import pymupdf

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PDF = OUTPUT_DIR / "sample_multimodal_report.pdf"


def generate_sample_pdf():
    print(f"Generating sample multimodal PDF at: {OUTPUT_PDF}")

    # 1. Create a matplotlib benchmark chart
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=200)
    models = ["Llama-3.1-8B", "Llama-3.2-11B-Vision", "GPT-4o-mini", "Claude-3.5-Haiku"]
    latency_ms = [45, 68, 120, 95]
    colors = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]

    bars = ax.bar(models, latency_ms, color=colors, width=0.55)
    ax.set_ylabel("Inference Latency (ms)", fontsize=11, fontweight="bold")
    ax.set_title("VLM Latency Benchmark on Groq LPU (Batch Size 1)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 2, f"{yval}ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()

    # Save chart to bytes
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close(fig)
    img_buf.seek(0)
    chart_bytes = img_buf.read()

    # 2. Build PDF using PyMuPDF
    doc = pymupdf.open()

    # Page 1: Overview & Benchmark Table
    page1 = doc.new_page(width=595, height=842)  # Standard A4
    page1.insert_text(
        (50, 60),
        "Technical Research Report: Next-Gen Multimodal RAG Architectures",
        fontsize=16,
        fontname="helv",
    )
    narrative_p1 = (
        "1. Executive Summary\n\n"
        "Modern enterprise Retrieval-Augmented Generation (RAG) systems frequently fail when processing\n"
        "unstructured documents containing heterogeneous modalities. Traditional naive chunking converts\n"
        "all pages into sequential text strings, destroying tabular relationships and stripping away charts,\n"
        "diagrams, and visual evidence.\n\n"
        "To solve this, our Multimodal RAG architecture employs layout-aware extraction with PyMuPDF,\n"
        "Vision-Language Model (VLM) enrichment via Groq's Llama 3.2 11B Vision, and hybrid multi-vector\n"
        "retrieval that pairs compact semantic summaries with raw high-resolution visual assets.\n\n"
        "2. Architecture Performance Comparison (Table 1)\n\n"
        "Below are the verified benchmark comparisons across standard retrieval evaluation metrics:"
    )
    page1.insert_text((50, 90), narrative_p1, fontsize=10, fontname="helv")

    # Draw Table on Page 1
    table_text = (
        "+--------------------------------+-----------------+-------------------+-----------------+\n"
        "| Architecture Type              | Context Recall  | Table Accuracy    | Vision Grounded |\n"
        "+--------------------------------+-----------------+-------------------+-----------------+\n"
        "| Naive Character Chunking       | 54.2%           | 21.0%             | 0.0%            |\n"
        "| Page-Level Text-Only RAG       | 68.5%           | 46.8%             | 0.0%            |\n"
        "| Multi-Vector Multimodal (Ours) | 94.7%           | 92.4%             | 96.1%           |\n"
        "+--------------------------------+-----------------+-------------------+-----------------+"
    )
    page1.insert_text((50, 270), table_text, fontsize=9, fontname="courier")

    p1_concl = (
        "\nKey Observation: As demonstrated in Table 1, the Multi-Vector Multimodal architecture achieves\n"
        "a 94.7% context recall and 96.1% visual groundedness score, dramatically outperforming naive methods."
    )
    page1.insert_text((50, 360), p1_concl, fontsize=10, fontname="helv")

    # Page 2: Chart & Visual Benchmark
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text(
        (50, 60),
        "3. Real-Time Hardware Latency Benchmarking (Figure 1)",
        fontsize=14,
        fontname="helv",
    )
    p2_narrative = (
        "Figure 1 below details the end-to-end token latency across several state-of-the-art vision models.\n"
        "When hosted on Groq LPU acceleration, Llama-3.2-11B-Vision achieves an average latency of 68ms,\n"
        "making it suitable for interactive conversational document analysis."
    )
    page2.insert_text((50, 90), p2_narrative, fontsize=10, fontname="helv")

    # Insert the chart image into Page 2
    chart_rect = pymupdf.Rect(50, 150, 545, 450)
    page2.insert_image(chart_rect, stream=chart_bytes)

    p2_caption = (
        "Figure 1: Comparison of inference latency for modern multimodal models on Groq LPUs.\n"
        "Conclusion: Llama-3.2-11B-Vision delivers the optimal trade-off of high visual precision and sub-70ms speed."
    )
    page2.insert_text((50, 470), p2_caption, fontsize=10, fontname="helv")

    doc.save(str(OUTPUT_PDF))
    doc.close()
    print("Sample PDF generated successfully!")


if __name__ == "__main__":
    generate_sample_pdf()
