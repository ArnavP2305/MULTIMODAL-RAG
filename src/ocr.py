"""
ocr.py — Tesseract OCR Engine for Image & Chart Text Extraction
Extracts verbatim text, numbers, axis labels, and diagram node strings from images.
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

# Auto-detect Tesseract binary path on Windows / Linux
POSSIBLE_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    shutil.which("tesseract"),
]

TESSERACT_EXECUTABLE = None
for p in POSSIBLE_TESSERACT_PATHS:
    if p and os.path.exists(p):
        TESSERACT_EXECUTABLE = p
        pytesseract.pytesseract.tesseract_cmd = p
        break


class OCROptimizer:
    """Preprocesses images and extracts verbatim text using Tesseract OCR."""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            self.tesseract_cmd = tesseract_cmd
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            self.tesseract_cmd = TESSERACT_EXECUTABLE

    def is_available(self) -> bool:
        """Check if Tesseract binary is accessible."""
        return self.tesseract_cmd is not None and os.path.exists(self.tesseract_cmd)

    def preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Enhance image contrast and convert to grayscale for higher OCR accuracy."""
        # Convert to grayscale
        gray = image.convert("L")
        # Increase contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.8)
        return enhanced

    def extract_text(self, image_path: str, psm: int = 3) -> str:
        """
        Extract raw text from an image or chart file.
        psm=3: Fully automatic page segmentation (default)
        psm=6: Assume a single uniform block of text
        psm=11: Sparse text / find as much text as possible
        """
        if not self.is_available() or not os.path.exists(image_path):
            return ""

        try:
            with Image.open(image_path) as img:
                # Preprocess
                processed_img = self.preprocess_image_for_ocr(img)
                # Tesseract config: preserve numbers and punctuation
                config = f"--psm {psm} --oem 3"
                text = pytesseract.image_to_string(processed_img, config=config)
                # Clean up whitespace
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                return "\n".join(lines)
        except Exception as e:
            print(f"[OCR Warning] Failed to OCR {image_path}: {e}")
            return ""

    def extract_detailed_ocr(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text along with word-level bounding boxes and confidence scores.
        """
        if not self.is_available() or not os.path.exists(image_path):
            return {"text": "", "words": [], "confidence": 0.0}

        try:
            with Image.open(image_path) as img:
                processed_img = self.preprocess_image_for_ocr(img)
                data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)

                words = []
                confidences = []
                for i in range(len(data["text"])):
                    w = data["text"][i].strip()
                    conf = float(data["conf"][i])
                    if w and conf > 30:  # Filter out low-confidence noise
                        words.append({
                            "word": w,
                            "bbox": (data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
                            "confidence": conf
                        })
                        confidences.append(conf)

                full_text = " ".join([w["word"] for w in words])
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

                return {
                    "text": full_text,
                    "words": words,
                    "avg_confidence": round(avg_conf, 2)
                }
        except Exception as e:
            return {"text": "", "words": [], "confidence": 0.0, "error": str(e)}


# Singleton instance
ocr_engine = OCROptimizer()
