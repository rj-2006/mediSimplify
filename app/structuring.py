"""
The core product: turn the merged, cross-checked transcription into a
plain-language summary + action checklist + warning signs.

Scope discipline (see conversation with Rahul before writing this):
- Describe what a drug is for and why it's prescribed -- fine.
- Never suggest dosage changes, whether to stop/start something, or any
  clinical judgment call. That's the line this prompt must not cross.
- Any low-confidence extraction (flagged by cross_check.py) must show up
  to the user as an explicit "please verify" item, not get smoothed over
  into a confident-sounding sentence.
"""
import json
from typing import List

from app import get_client, get_model, clean_json_text
from app.models import ActionItem, CrossCheckResult, StructuredOutput

STRUCTURING_PROMPT = """You are MediSimplify, a health-literacy translator.
You take a transcribed medical document (prescription, lab report, or
discharge summary) and turn it into a plain-language explanation for a
patient with low health literacy, who may be elderly, a non-native
speaker, or unfamiliar with medical terms.

STRICT RULES -- do not break these:
1. Describe what medications/tests are FOR and WHY they were likely
   prescribed/ordered. Never suggest dosage changes, never tell the user
   to start, stop, skip, or adjust anything beyond restating exactly
   what is written in the document.
2. Use short sentences, everyday words (grade 6-8 reading level).
3. Never invent information that isn't in the document. If something is
   unclear or marked [ILLEGIBLE], say so plainly and add it to
   low_confidence_flags instead of guessing.
4. Warning signs should only include symptoms that would warrant
   contacting a doctor/going to the ER -- based on what's standard for
   the specific medications/conditions mentioned, not generic filler.
5. If overall extraction confidence is "low" or "medium", say so in the
   summary itself, don't bury it only in the flags list.

--- Document transcription (confidence: {confidence}) ---
{merged_text}

--- Known extraction disagreements (treat these as uncertain) ---
{disagreements}

Respond with ONLY valid JSON, no markdown fences, no preamble, matching
this exact schema:
{{
  "plain_summary": "2-4 sentences, what this document is and says",
  "what_it_means_for_you": "2-4 sentences, practical meaning for the patient",
  "action_checklist": [
    {{"text": "string", "is_warning_sign": false}}
  ],
  "warning_signs": ["string", ...],
  "low_confidence_flags": ["string", ...]
}}"""


def _format_disagreements(disagreements) -> str:
    if not disagreements:
        return "(none)"
    lines = []
    for d in disagreements:
        lines.append(
            f"- {d.field_type}: OCR read '{d.paddle_value}', "
            f"vision model read '{d.vision_value}' -- unconfirmed"
        )
    return "\n".join(lines)


def structure_document(cross_check_result: CrossCheckResult) -> StructuredOutput:
    client = get_client()
    model = get_model()
    prompt = STRUCTURING_PROMPT.format(
        confidence=cross_check_result.overall_confidence,
        merged_text=cross_check_result.merged_text,
        disagreements=_format_disagreements(cross_check_result.disagreements),
    )

    response = client.chat.completions.create(
        model=model,
        max_tokens=1500,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    raw = clean_json_text(response.choices[0].message.content or "")

    try:
        parsed = json.loads(raw)
        action_checklist: List[ActionItem] = [
            ActionItem(**item) if isinstance(item, dict) else ActionItem(text=str(item))
            for item in parsed.get("action_checklist", [])
        ]
        return StructuredOutput(
            plain_summary=parsed.get("plain_summary", ""),
            what_it_means_for_you=parsed.get("what_it_means_for_you", ""),
            action_checklist=action_checklist,
            warning_signs=[str(w) for w in parsed.get("warning_signs", [])],
            low_confidence_flags=[str(f) for f in parsed.get("low_confidence_flags", [])],
        )
    except Exception as e:
        # Fail loud-but-safe: don't show a confident-looking summary built
        # from a parse failure. Surface the raw model output as a flag.
        return StructuredOutput(
            plain_summary=(
                "We couldn't reliably process this document. Please show "
                "the original to your pharmacist or doctor."
            ),
            what_it_means_for_you="Processing failed -- no summary available.",
            action_checklist=[],
            warning_signs=[],
            low_confidence_flags=[f"Automatic structuring failed to parse: {str(e)[:60]}"],
        )

