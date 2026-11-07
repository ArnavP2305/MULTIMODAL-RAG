"""
summarizer.py — Multimodal Element Summarizer with Groq Vision & OCR Fusion
Fuses Tesseract OCR text with Groq Llama 3.2 11B Vision to generate dense,
factually verified descriptions of charts, diagrams, tables, and scanned pages.
"""
import os
import base64
from pathlib import Path
from typing import Optional
from groq import Groq

from .config import GROQ_API_KEY, GROQ_VISION_MODEL, GROQ_CHAT_MODEL


class GroqMultimodalSummarizer:
    """Summarizes images, charts, and tables using Groq Vision fused with OCR text."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GROQ_API_KEY
        self.client = None
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"[Warning] Failed to initialize Groq client: {e}")

    def _encode_image(self, image_path: str) -> str:
        """Encode local image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def summarize_image(self, image_path: str, ocr_text: str = "", context: str = "") -> str:
        """Analyze an extracted image or chart using Groq Vision model + OCR ground truth."""
        fallback = f"Visual figure/image: {Path(image_path).name}."
        if ocr_text:
            fallback += f" Text detected inside image via OCR:\n{ocr_text}"

        if not self.client or not os.path.exists(image_path):
            return fallback

        try:
            base64_image = self._encode_image(image_path)
            prompt = (
                "You are an expert multimodal document analyst.\n"
                "Examine this visual element (chart, graph, diagram, architecture flow, or photo) extracted from a document.\n"
                "Provide a clear, detailed, and factually rigorous description:\n"
                "1. Visual type (e.g., bar chart, line graph, system architecture, flow diagram).\n"
                "2. Specific data points: exact numbers, axis labels, legends, units, and categories.\n"
                "3. Core insight: what conclusion or trend does this visual convey?\n"
            )

            # Inject verbatim OCR text extracted from the image
            if ocr_text and ocr_text.strip():
                prompt += (
                    f"\n[GROUND-TRUTH OCR TEXT EXTRACTED DIRECTLY FROM THIS IMAGE VIA TESSERACT]:\n"
                    f"{ocr_text.strip()}\n"
                    f"Use the exact names, numbers, and labels from the OCR text above to ensure 100% precision.\n"
                )

            if context:
                prompt += f"\nSurrounding page context:\n{context}\n"

            try:
                response = self.client.chat.completions.create(
                    model=GROQ_VISION_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=600,
                    temperature=0.1,
                )
            except Exception as v_err:
                if "decommissioned" in str(v_err) or "llama-3.2" in str(v_err):
                    response = self.client.chat.completions.create(
                        model="qwen/qwen3.8-27b",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{base64_image}"
                                        },
                                    },
                                ],
                            }
                        ],
                        max_tokens=600,
                        temperature=0.1,
                    )
                else:
                    raise v_err
            vlm_caption = response.choices[0].message.content.strip()

            # Append raw OCR text for guaranteed token indexing in hybrid search
            if ocr_text and ocr_text.strip():
                return f"{vlm_caption}\n\n[OCR Extracted Text]:\n{ocr_text.strip()}"
            return vlm_caption

        except Exception as e:
            return f"Visual asset ({Path(image_path).name}): {str(e)}\n\n[OCR Extracted Text]:\n{ocr_text}"

    def summarize_table(self, table_markdown: str, page_num: int = 1, ocr_text: str = "") -> str:
        """Generate a semantic summary of a structured table."""
        if not table_markdown or not table_markdown.strip():
            table_markdown = ocr_text or f"Table on page {page_num}"

        if not self.client:
            first_rows = "\n".join(table_markdown.strip().split("\n")[:4])
            return f"Structured table on page {page_num}.\nSample data:\n{first_rows}"

        try:
            prompt = (
                f"You are a financial and technical data analyst.\n"
                f"Analyze the following table extracted from page {page_num} of a document:\n\n"
                f"{table_markdown}\n\n"
                f"Generate a concise, dense semantic summary:\n"
                f"- What metrics, entities, or categories does this table represent?\n"
                f"- What are the most significant values, totals, peaks, or outliers?\n"
                f"- Highlight any key comparisons between columns or time periods."
            )

            response = self.client.chat.completions.create(
                model=GROQ_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            first_rows = "\n".join(table_markdown.strip().split("\n")[:4])
            return f"Table on page {page_num}. Data:\n{first_rows}"

    def transcribe_scanned_page(self, page_image_path: str, ocr_text: str = "") -> str:
        """Perform OCR and visual description on a scanned page image."""
        if not self.client or not os.path.exists(page_image_path):
            return f"Scanned page at {Path(page_image_path).name}.\nOCR Text:\n{ocr_text}"

        try:
            base64_image = self._encode_image(page_image_path)
            prompt = (
                "You are an expert OCR and document analysis engine.\n"
                "This is a high-resolution image of a scanned page.\n"
                "1. Transcribe all legible text verbatim with proper layout structure.\n"
                "2. Note any stamps, signatures, hand-written notes, logos, or embedded charts.\n"
            )
            if ocr_text:
                prompt += f"\nInitial Tesseract OCR transcript:\n{ocr_text}\n"

            response = self.client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Scanned page ({Path(page_image_path).name}): {str(e)}\n\nOCR Text:\n{ocr_text}"
