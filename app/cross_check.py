"""
Reconciliation engine: cross-checks independent OCR and Vision LLM extractions
to detect medically significant discrepancies and compute confidence metrics.
"""
import json
import re
from typing import List

from app import get_client, get_model, clean_json_text
from app.models import CrossCheckResult, TokenDisagreement

DOSAGE_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s?(mg|mcg|ml|g|units?|iu|tabs?|tablets?|caps?)\b",
    re.IGNORECASE,
)

RECONCILE_PROMPT = """You are given two independent OCR transcriptions of the SAME medical document image. One is from a traditional OCR engine (PaddleOCR), the other from a vision-capable LLM. They may disagree in places, especially on handwritten drug names, doses, or numbers.

--- PaddleOCR transcription ---
{paddle_text}

--- Vision LLM transcription ---
{vision_text}

Do two things:
1. Produce a single best-guess merged transcription. Where both agree, use that. Where they disagree, pick the more plausible medical reading, but if truly unclear, keep both as "A / B".
2. List every point where the two transcriptions disagree on a drug name, dosage, frequency, or number. Do NOT list disagreements in whitespace, casing, or formatting — only medically meaningful ones.

Respond with ONLY valid JSON matching this exact schema:
{{
  "merged_text": "string",
  "disagreements": [
    {{"field_type": "drug_name|dosage|frequency|number|other",
      "paddle_value": "string",
      "vision_value": "string"}}
  ],
  "overall_confidence": "high|medium|low"
}}

Use "low" confidence if there are 2+ medically meaningful disagreements, "medium" for exactly 1, "high" for none."""


def _regex_dosage_tokens(text: str) -> List[str]:
    """Extracts standardized dosage and numerical unit tokens via regular expressions."""
    return [m.group(0).lower().strip() for m in DOSAGE_PATTERN.finditer(text)]


def cross_check(paddle_text: str, vision_text: str) -> CrossCheckResult:
    """Reconciles PaddleOCR and Vision LLM extractions into a unified text with confidence scoring."""
    client = get_client()
    model = get_model()

    paddle_tokens = set(_regex_dosage_tokens(paddle_text))
    vision_tokens = set(_regex_dosage_tokens(vision_text))
    token_mismatch = paddle_tokens != vision_tokens

    try:
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=4096,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": RECONCILE_PROMPT.format(
                        paddle_text=paddle_text or "(empty)",
                        vision_text=vision_text or "(empty)",
                    ),
                }
            ],
            response_format={"type": "json_object"},
        )

        raw = clean_json_text(response.choices[0].message.content or "")
        parsed = json.loads(raw)
        disagreements = [
            TokenDisagreement(**d) for d in parsed.get("disagreements", [])
        ]
        merged_text = parsed.get("merged_text", vision_text)
        overall_confidence = parsed.get("overall_confidence", "medium")
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "quota" in err_msg.lower() or "exceeded" in err_msg.lower():
            merged_text = f"API Rate Limit Exceeded (Google Gemini Free Tier limit reached).\n\n[PaddleOCR Fallback]:\n{paddle_text}"
        else:
            merged_text = vision_text or paddle_text
        disagreements = []
        overall_confidence = "low"

    if token_mismatch and overall_confidence == "high":
        overall_confidence = "medium"

    return CrossCheckResult(
        paddle_text=paddle_text,
        vision_text=vision_text,
        merged_text=merged_text,
        disagreements=disagreements,
        overall_confidence=overall_confidence,
    )
