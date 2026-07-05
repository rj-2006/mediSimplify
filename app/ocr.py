"""
PaddleOCR text extraction engine and PDF rasterization utilities.
Provides deterministic local OCR capabilities for printed text and tables.
"""
import tempfile
from functools import lru_cache
from typing import List

from app.models import OCRResult


@lru_cache(maxsize=1)
def _get_paddle_engine():
    """Lazy initialization and caching of the PaddleOCR inference engine."""
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def run_paddle_ocr(image_path: str) -> OCRResult:
    """Executes PaddleOCR on the target image and returns concatenated text with average confidence."""
    engine = _get_paddle_engine()
    result = engine.ocr(image_path, cls=True)

    lines: List[str] = []
    confidences: List[float] = []

    for page in result:
        if not page:
            continue
        for detection in page:
            _box, (text, conf) = detection
            lines.append(text)
            confidences.append(conf)

    raw_text = "\n".join(lines)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return OCRResult(source="paddleocr", raw_text=raw_text, confidence=avg_conf)


def prepare_image_for_ocr(input_path: str) -> str:
    """Rasterizes multi-page PDF documents into a single stitched PNG image for full pipeline extraction."""
    if input_path.lower().endswith(".pdf"):
        from pdf2image import convert_from_path
        from PIL import Image

        pages = convert_from_path(input_path, dpi=300)
        if len(pages) > 1:
            total_height = sum(p.height for p in pages)
            max_width = max(p.width for p in pages)
            combined = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))
            y_offset = 0
            for page in pages:
                combined.paste(page, (0, y_offset))
                y_offset += page.height
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            combined.save(tmp.name)
            return tmp.name
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            pages[0].save(tmp.name)
            return tmp.name
    return input_path
