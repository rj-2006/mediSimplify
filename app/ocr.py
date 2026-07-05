"""
PaddleOCR extraction path.

This is the "dedicated OCR model" side of the dual-extraction cross-check.
It's fast, local, deterministic, and generally strong on printed text and
tabular lab-report layouts. It's weak on handwriting -- that's what the
vision LLM path (vision_extract.py) is for.
"""
from functools import lru_cache
from typing import List, Tuple

from app.models import OCRResult


@lru_cache(maxsize=1)
def _get_paddle_engine():
    # Lazy import + cache: PaddleOCR is slow to initialize (loads model
    # weights), so we only want to pay that cost once per process, not
    # once per request.
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def run_paddle_ocr(image_path: str) -> OCRResult:
    """Run PaddleOCR on a single image and return concatenated text plus
    an averaged confidence score.

    PaddleOCR 2.x returns results per detected text line as:
        [[box_points], (text, confidence)]
    We just need the text (in reading order, top-to-bottom) and an
    average confidence to inform the cross-check step.
    """
    engine = _get_paddle_engine()
    result = engine.ocr(image_path, cls=True)

    lines: List[str] = []
    confidences: List[float] = []

    # result is a list (one per image) of lists of detections
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
    """If given a PDF, rasterize ALL pages and stitch them vertically into
    a single temp PNG. Multi-page discharge summaries were previously
    silently truncated at page 1 -- this fixes that.
    If already an image, return as-is."""
    if input_path.lower().endswith(".pdf"):
        from pdf2image import convert_from_path
        import tempfile
        from PIL import Image

        pages = convert_from_path(input_path, dpi=300)
        if len(pages) > 1:
            # Stitch all pages vertically so OCR + vision see the full doc.
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
