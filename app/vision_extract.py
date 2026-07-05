"""
Vision-LLM extraction path.

This is the side that handles handwriting, messy layouts, stamps,
letterheads, and abbreviation-splitting better than a dedicated OCR model.
It trades determinism for robustness -- exactly the tradeoff we want as a
*second* opinion, not the only one.
"""
import base64
import mimetypes

from app import get_client, get_model
from app.models import OCRResult

EXTRACTION_PROMPT = """You are extracting text from a photo of a medical
document (prescription, lab report, or discharge summary).

Transcribe ALL visible text exactly as written, preserving:
- Drug names and dosages exactly as they appear, even if abbreviated
- Numbers exactly as written (do not "correct" what looks like a typo)
- Frequency/timing abbreviations (BID, PRN, QID, etc.) exactly as written
- Table structure from lab reports as plain text rows

If any word or number is genuinely illegible, write [ILLEGIBLE] in its
place rather than guessing. Do not add any commentary, explanation, or
formatting beyond the transcription itself. Output only the transcribed
text."""


def run_vision_extraction(image_path: str) -> OCRResult:
    client = get_client()
    model = get_model()
    media_type, _ = mimetypes.guess_type(image_path)
    if media_type is None:
        media_type = "image/jpeg"

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model=model,
        max_tokens=2000,
        temperature=0.1,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
                        },
                    },
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ],
    )

    raw_text = response.choices[0].message.content or ""
    return OCRResult(source="vision_llm", raw_text=raw_text, confidence=None)
