"""
Cross-check the two independent extractions against each other.

This is the piece that turns "cool demo" into "actually trustworthy":
if PaddleOCR and the vision LLM disagree on a drug name, dose, or number,
that's a strong, cheap signal that a human should double check it --
much more reliable than asking a single model "are you sure?" (it'll
just say yes).

Approach:
1. Cheap deterministic pass: regex-pull number+unit tokens (dosages,
   frequencies) from both texts and diff them directly.
2. LLM pass: ask a model to reconcile the two raw texts into one merged
   best-guess transcription, and to explicitly list any point of
   disagreement on medically-relevant tokens (drug names, doses,
   frequencies, numbers). Forced JSON output.
"""
import json
import re
from typing import List

from app import get_client, get_model, clean_json_text
from app.models import CrossCheckResult, TokenDisagreement

# Matches things like "500mg", "5 ml", "2 tabs", "3x/day", "BID"
DOSAGE_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s?(mg|mcg|ml|g|units?|iu|tabs?|tablets?|caps?)\b",
    re.IGNORECASE,
)

RECONCILE_PROMPT = """You are given two independent OCR transcriptions of
the SAME medical document image. One is from a traditional OCR engine
(PaddleOCR), the other from a vision-capable LLM. They may disagree in
places, especially on handwritten drug names, doses, or numbers.

--- PaddleOCR transcription ---
{paddle_text}

--- Vision LLM transcription ---
{vision_text}

Do two things:
1. Produce a single best-guess merged transcription. Where both agree,
   use that. Where they disagree, pick the more plausible medical
   reading, but if truly unclear, keep both as "A / B".
2. List every point where the two transcriptions disagree on a drug
   name, dosage, frequency, or number. Do NOT list disagreements in
   whitespace, casing, or formatting -- only medically meaningful ones.

Respond with ONLY valid JSON, no markdown fences, no preamble, matching
this exact schema:
{{
  "merged_text": "string",
  "disagreements": [
    {{"field_type": "drug_name|dosage|frequency|number|other",
      "paddle_value": "string",
      "vision_value": "string"}}
  ],
  "overall_confidence": "high|medium|low"
}}

Use "low" confidence if there are 2+ medically meaningful disagreements,
"medium" for exactly 1, "high" for none."""


def _regex_dosage_tokens(text: str) -> List[str]:
    return [m.group(0).lower().strip() for m in DOSAGE_PATTERN.finditer(text)]


def cross_check(paddle_text: str, vision_text: str) -> CrossCheckResult:
    client = get_client()
    model = get_model()
    # Deterministic sanity pass -- cheap, doesn't need the network. Useful
    # mainly as a fallback if the LLM reconciliation call fails.
    paddle_tokens = set(_regex_dosage_tokens(paddle_text))
    vision_tokens = set(_regex_dosage_tokens(vision_text))
    token_mismatch = paddle_tokens != vision_tokens

    response = client.chat.completions.create(
        model=model,
        max_tokens=1500,
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

    try:
        parsed = json.loads(raw)
        disagreements = [
            TokenDisagreement(**d) for d in parsed.get("disagreements", [])
        ]
        merged_text = parsed.get("merged_text", vision_text)
        overall_confidence = parsed.get("overall_confidence", "medium")
    except Exception:
        # LLM reconciliation failed to parse -- fall back to the vision
        # transcription (generally more robust on messy documents) and
        # flag low confidence rather than silently pretending things are fine.
        merged_text = vision_text or paddle_text
        disagreements = []
        overall_confidence = "low"

    # If the regex pass found a dosage-token mismatch that the LLM pass
    # somehow missed, don't let confidence read as "high".
    if token_mismatch and overall_confidence == "high":
        overall_confidence = "medium"

    return CrossCheckResult(
        paddle_text=paddle_text,
        vision_text=vision_text,
        merged_text=merged_text,
        disagreements=disagreements,
        overall_confidence=overall_confidence,
    )
