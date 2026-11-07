"""
extractor.py — Multimodal PDF Layout & Asset Extractor with OCR Integration
Extracts Text, Tables, Charts, Embedded Images, and Scanned Pages using PyMuPDF and Tesseract OCR.
"""
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import pymupdf  # PyMuPDF
from PIL import Image
import io

from .config import IMAGES_DIR, TABLES_DIR
from .ocr import ocr_engine


class ExtractedElement:
    """Represents an extracted multimodal element from a document."""

    def __init__(
        self,
        element_id: str,
        element_type: str,  # 'text', 'table', 'image', 'scanned_page'
        page_number: int,
        source_doc: str,
        content: str = "",
        image_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.element_id = element_id
        self.element_type = element_type
        self.page_number = page_number
        self.source_doc = source_doc
        self.content = content
        self.image_path = image_path
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "page_number": self.page_number,
            "source_doc": self.source_doc,
            "content": self.content,
            "image_path": self.image_path,
            "metadata": self.metadata,
        }


class MultimodalPDFExtractor:
    """Extracts text chunks, markdown tables, and charts/images with integrated OCR."""

    def __init__(
        self,
        min_image_width: int = 100,
        min_image_height: int = 100,
        scanned_text_threshold: int = 60,
    ):
        self.min_image_width = min_image_width
        self.min_image_height = min_image_height
        self.scanned_text_threshold = scanned_text_threshold
        self.ocr = ocr_engine

    def extract(self, pdf_path: str) -> List[ExtractedElement]:
        """Process a PDF and return a list of multimodal elements with OCR metadata."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        doc_name = path.name
        doc = pymupdf.open(str(path))
        elements: List[ExtractedElement] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1

            # 1. Check for Tables
            table_bboxes = []
            try:
                tables = page.find_tables()
                for t_idx, table in enumerate(tables):
                    table_bboxes.append(table.bbox)
                    # Convert table to markdown
                    try:
                        df = table.extract()
                        if df and len(df) > 0:
                            cleaned_headers = [str(h).strip() if h is not None else "" for h in df[0]]
                            md_lines = ["| " + " | ".join(cleaned_headers) + " |"]
                            md_lines.append("| " + " | ".join(["---"] * len(cleaned_headers)) + " |")
                            for row in df[1:]:
                                cleaned_row = [str(c).strip() if c is not None else "" for c in row]
                                md_lines.append("| " + " | ".join(cleaned_row) + " |")
                            table_md = "\n".join(md_lines)
                        else:
                            table_md = ""
                    except Exception:
                        table_md = ""

                    # Render cropped image of the table
                    table_rect = pymupdf.Rect(table.bbox)
                    pix = page.get_pixmap(clip=table_rect, dpi=200)
                    table_img_filename = f"{path.stem}_p{page_num}_table_{t_idx}.png"
                    table_img_path = str(TABLES_DIR / table_img_filename)
                    pix.save(table_img_path)

                    # Run OCR on table image as backup
                    table_ocr_text = self.ocr.extract_text(table_img_path)

                    elements.append(
                        ExtractedElement(
                            element_id=f"{path.stem}_p{page_num}_tbl_{t_idx}",
                            element_type="table",
                            page_number=page_num,
                            source_doc=doc_name,
                            content=table_md or table_ocr_text,
                            image_path=table_img_path,
                            metadata={
                                "bbox": table.bbox,
                                "table_index": t_idx,
                                "ocr_text": table_ocr_text,
                            },
                        )
                    )
            except Exception:
                pass

            # 2. Extract Embedded Images & Visual Figures with OCR
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_img = doc.extract_image(xref)
                img_bytes = base_img["image"]
                img_ext = base_img["ext"]
                width = base_img["width"]
                height = base_img["height"]

                # Filter out tiny icons, logos
                if width < self.min_image_width or height < self.min_image_height:
                    continue

                img_filename = f"{path.stem}_p{page_num}_img_{img_idx}.{img_ext}"
                saved_img_path = str(IMAGES_DIR / img_filename)
                with open(saved_img_path, "wb") as f:
                    f.write(img_bytes)

                # Execute OCR directly on the image to read embedded text/labels
                image_ocr_text = self.ocr.extract_text(saved_img_path)

                elements.append(
                    ExtractedElement(
                        element_id=f"{path.stem}_p{page_num}_img_{img_idx}",
                        element_type="image",
                        page_number=page_num,
                        source_doc=doc_name,
                        content=image_ocr_text,  # Seeded with verbatim OCR text
                        image_path=saved_img_path,
                        metadata={
                            "width": width,
                            "height": height,
                            "format": img_ext,
                            "image_index": img_idx,
                            "ocr_text": image_ocr_text,
                        },
                    )
                )

            # 3. Extract Text Blocks (excluding table bounding boxes)
            text_page = page.get_text("blocks")
            accumulated_text = []

            for block in text_page:
                if block[6] == 0:  # Text block
                    b_bbox = block[:4]
                    b_text = block[4].strip()

                    # Avoid duplicate text inside tables
                    is_inside_table = False
                    for t_bbox in table_bboxes:
                        if (
                            b_bbox[0] >= t_bbox[0] - 5
                            and b_bbox[1] >= t_bbox[1] - 5
                            and b_bbox[2] <= t_bbox[2] + 5
                            and b_bbox[3] <= t_bbox[3] + 5
                        ):
                            is_inside_table = True
                            break

                    if not is_inside_table and b_text:
                        accumulated_text.append(b_text)

            total_page_text = "\n\n".join(accumulated_text)

            # 4. Check for Scanned / Image-only Pages
            if len(total_page_text.strip()) < self.scanned_text_threshold and len(image_list) == 0:
                pix = page.get_pixmap(dpi=200)
                scanned_img_filename = f"{path.stem}_p{page_num}_scanned.png"
                scanned_path = str(IMAGES_DIR / scanned_img_filename)
                pix.save(scanned_path)

                # Full page OCR using Tesseract
                scanned_ocr_text = self.ocr.extract_text(scanned_path)

                elements.append(
                    ExtractedElement(
                        element_id=f"{path.stem}_p{page_num}_scanned",
                        element_type="scanned_page",
                        page_number=page_num,
                        source_doc=doc_name,
                        content=scanned_ocr_text,
                        image_path=scanned_path,
                        metadata={"is_scanned": True, "ocr_text": scanned_ocr_text},
                    )
                )
            elif total_page_text:
                chunks = self._chunk_text(total_page_text, max_chunk_size=1200, overlap=150)
                for c_idx, chunk in enumerate(chunks):
                    elements.append(
                        ExtractedElement(
                            element_id=f"{path.stem}_p{page_num}_txt_{c_idx}",
                            element_type="text",
                            page_number=page_num,
                            source_doc=doc_name,
                            content=chunk,
                            image_path=None,
                            metadata={"chunk_index": c_idx},
                        )
                    )

        doc.close()
        return elements

    def _chunk_text(self, text: str, max_chunk_size: int = 1200, overlap: int = 150) -> List[str]:
        if len(text) <= max_chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chunk_size
            if end >= len(text):
                chunks.append(text[start:].strip())
                break
            split_at = text.rfind("\n\n", start, end)
            if split_at == -1:
                split_at = text.rfind(". ", start, end)
            if split_at == -1 or split_at < start + (max_chunk_size // 2):
                split_at = end
            else:
                split_at += 1

            chunk_content = text[start:split_at].strip()
            if chunk_content:
                chunks.append(chunk_content)
            start = max(start + 1, split_at - overlap)

        return chunks
