"""
Secondary translation engine: translates patient-facing structured health summaries
into target languages while preserving clinical safety rules and medication names.
"""
import json
from app import get_client, get_model, clean_json_text
from app.models import StructuredOutput

TRANSLATE_PROMPT = """Translate the following patient-facing health summary into {language}. Keep it at a simple, plain-language reading level (grade 6-8). Translate the medical meanings clearly.

Do NOT translate:
- Drug names or brand names
- Specific dosages or numbers
- Medical warning signs that rely on specific terms
- The overall_status value (keep as "NORMAL", "ATTENTION", "URGENT", or "UNCONFIRMED")
- The status field inside key_findings (keep as "normal", "attention", "urgent", or "unconfirmed")

--- English Structured Output ---
{content}

Respond with ONLY valid JSON matching the exact same schema:
{{
  "overall_status": "string",
  "plain_summary": "string",
  "what_it_means_for_you": "string",
  "key_findings": [
    {{
      "item_name": "string",
      "value_or_dose": "string",
      "reference_or_purpose": "string",
      "status": "string",
      "explanation": "string"
    }}
  ],
  "action_checklist": [{{"text": "string", "is_warning_sign": false}}],
  "warning_signs": ["string"],
  "low_confidence_flags": ["string"],
  "disclaimer": "string"
}}"""


def translate_structured_output(
    structured: StructuredOutput, target_language: str
) -> StructuredOutput:
    """Translates structured patient summary into the requested language."""
    try:
        client = get_client()
        model = get_model()

        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=4096,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": TRANSLATE_PROMPT.format(
                        language=target_language,
                        content=structured.model_dump_json(indent=2),
                    ),
                }
            ],
            response_format={"type": "json_object"},
        )
        raw = clean_json_text(response.choices[0].message.content or "")
        parsed = json.loads(raw)
        return StructuredOutput(**parsed)
    except Exception as e:
        err_msg = str(e)
        fallback = structured.model_copy()
        if "429" in err_msg or "quota" in err_msg.lower() or "exceeded" in err_msg.lower():
            fallback.low_confidence_flags = list(structured.low_confidence_flags) + [
                f"Translation to {target_language} failed due to Google Gemini Free Tier API rate limit (429). Showing original English."
            ]
        else:
            fallback.low_confidence_flags = list(structured.low_confidence_flags) + [
                f"Translation to {target_language} failed to parse ({str(e)[:60]}). Showing original English."
            ]
        return fallback
