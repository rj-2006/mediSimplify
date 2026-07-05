"""
Vision LLM text extraction engine.
Provides robust multimodal transcription for complex layouts, stamps, and handwritten medical notes.
"""
import base64
import mimetypes

from app import get_client, get_model
from app.models import OCRResult

EXTRACTION_PROMPT = """You are extracting text from a photo of a medical document (prescription, lab report, or discharge summary).

Transcribe ALL visible text exactly as written, preserving:
- Drug names and dosages exactly as they appear, even if abbreviated
- Numbers exactly as written (do not correct what appears to be a typo)
- Frequency/timing abbreviations (BID, PRN, QID, etc.) exactly as written
- Table structure from lab reports as plain text rows

If any word or number is genuinely illegible, write [ILLEGIBLE] in its place rather than guessing. Do not add any commentary, explanation, or formatting beyond the transcription itself. Output only the transcribed text."""


def run_vision_extraction(image_path: str) -> OCRResult:
    """Executes vision-based multimodal transcription on the target image."""
    try:
        client = get_client()
        model = get_model()
        media_type, _ = mimetypes.guess_type(image_path)
        if media_type is None:
            media_type = "image/jpeg"

        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=4096,
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
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "quota" in err_msg.lower() or "exceeded" in err_msg.lower():
            err_str = "[Vision LLM Error: API Rate Limit Exceeded (Google Gemini Free Tier limit reached)]"
        else:
            err_str = f"[Vision LLM Error: {str(e)[:100]}]"
        return OCRResult(source="vision_llm", raw_text=err_str, confidence=0.0)
