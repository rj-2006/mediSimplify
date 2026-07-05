"""
Core structuring engine: transforms cross-checked medical transcriptions into
structured, plain-language patient summaries with clinical safety guardrails.
"""
import json
from typing import List

from app import get_client, get_model, clean_json_text
from app.models import ActionItem, CrossCheckResult, KeyFinding, StructuredOutput

STRUCTURING_PROMPT = """You are MediSimplify, an advanced health-literacy translation engine.
You take a transcribed medical document (prescription, lab report, or discharge summary) and turn it into a structured, plain-language explanation for a patient with low health literacy, who may be elderly, a non-native speaker, or unfamiliar with medical terms.

STRICT CLINICAL & SAFETY RULES:
1. Describe what medications/tests are FOR and WHY they were likely prescribed/ordered. Never suggest dosage changes, never tell the user to start, stop, skip, or adjust anything beyond restating exactly what is written in the document.
2. Use short sentences and everyday words (grade 6-8 reading level).
3. Never invent information that is not present in the document. If a word or value is unclear or marked [ILLEGIBLE], state so plainly and include it in low_confidence_flags rather than guessing.
4. Warning signs must only include emergency symptoms that would warrant contacting a doctor or going to the ER based on what is standard for the specific conditions/medications mentioned.
5. Set overall_status to:
   - "NORMAL" if all test results/medications appear routine with no alarming values.
   - "ATTENTION" if any lab value is outside reference ranges or requires medical follow-up.
   - "URGENT" if critical lab values or acute emergency warnings are noted.
   - "UNCONFIRMED" if extraction confidence is low or major discrepancies exist.
6. Extract key_findings as a list of structured items for EVERY major medication prescribed or lab test reported.

--- Document Transcription (Confidence: {confidence}) ---
{merged_text}

--- Known OCR Discrepancies (Treat as Uncertain) ---
{disagreements}

Respond with ONLY valid JSON matching this exact schema:
{{
  "overall_status": "NORMAL | ATTENTION | URGENT | UNCONFIRMED",
  "plain_summary": "2-4 sentences summarizing what this document is and states.",
  "what_it_means_for_you": "2-4 sentences explaining practical next steps and meaning for the patient.",
  "key_findings": [
    {{
      "item_name": "Name of medication or lab test",
      "value_or_dose": "Dose, frequency, or measured test value",
      "reference_or_purpose": "Normal range (for labs) or primary purpose (for drugs)",
      "status": "normal | attention | urgent | unconfirmed",
      "explanation": "Simple 1-2 sentence explanation of this specific item."
    }}
  ],
  "action_checklist": [
    {{"text": "Concrete actionable instruction", "is_warning_sign": false}}
  ],
  "warning_signs": ["Symptom requiring urgent medical contact"],
  "low_confidence_flags": ["Unclear or illegible item requiring verification"]
}}"""


def _format_disagreements(disagreements) -> str:
    if not disagreements:
        return "(none)"
    lines = []
    for d in disagreements:
        lines.append(
            f"- {d.field_type}: OCR read '{d.paddle_value}', "
            f"vision model read '{d.vision_value}' — unconfirmed"
        )
    return "\n".join(lines)


def structure_document(cross_check_result: CrossCheckResult) -> StructuredOutput:
    """Passes reconciled text to LLM to generate plain-language JSON structure."""
    try:
        client = get_client()
        model = get_model()
        prompt = STRUCTURING_PROMPT.format(
            confidence=cross_check_result.overall_confidence,
            merged_text=cross_check_result.merged_text,
            disagreements=_format_disagreements(cross_check_result.disagreements),
        )

        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=4096,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        raw = clean_json_text(response.choices[0].message.content or "")
        parsed = json.loads(raw)
        action_checklist: List[ActionItem] = [
            ActionItem(**item) if isinstance(item, dict) else ActionItem(text=str(item))
            for item in parsed.get("action_checklist", [])
        ]
        key_findings: List[KeyFinding] = [
            KeyFinding(**item) if isinstance(item, dict) else KeyFinding(
                item_name=str(item),
                value_or_dose="",
                reference_or_purpose="",
                status="unconfirmed",
                explanation=""
            )
            for item in parsed.get("key_findings", [])
        ]
        return StructuredOutput(
            overall_status=parsed.get("overall_status", "NORMAL"),
            plain_summary=parsed.get("plain_summary", ""),
            what_it_means_for_you=parsed.get("what_it_means_for_you", ""),
            key_findings=key_findings,
            action_checklist=action_checklist,
            warning_signs=[str(w) for w in parsed.get("warning_signs", [])],
            low_confidence_flags=[str(f) for f in parsed.get("low_confidence_flags", [])],
        )
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "quota" in err_msg.lower() or "exceeded" in err_msg.lower() or "rate" in err_msg.lower():
            summary_msg = "We couldn't process this document because your Google Gemini Free Tier API rate limit (20 requests/day) was exceeded."
            meaning_msg = "Please wait 1 minute before retrying, or configure a different API key (OpenAI/Gemini/OpenRouter) in your .env file."
            flag_msg = "API Rate Limit Exceeded (429 Resource Exhausted)."
        else:
            summary_msg = "We couldn't reliably process this document. Please show the original to your pharmacist or doctor."
            meaning_msg = "Processing failed — no summary available."
            flag_msg = f"Automatic structuring failed to parse: {str(e)[:60]}"

        return StructuredOutput(
            overall_status="UNCONFIRMED",
            plain_summary=summary_msg,
            what_it_means_for_you=meaning_msg,
            key_findings=[],
            action_checklist=[],
            warning_signs=[],
            low_confidence_flags=[flag_msg],
        )
