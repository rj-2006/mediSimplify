"""
Optional second-language pass. Deliberately simple: translate the
already-structured, already-safety-checked output rather than
re-running the whole pipeline in another language. This way the
safety/scope rules in structuring.py only need to be enforced once.
"""
from app import get_client, get_model, clean_json_text
from app.models import StructuredOutput

TRANSLATE_PROMPT = """Translate the following patient-facing health
summary into {language}. Keep it at a simple, plain-language reading
level (grade 6-8). Translate the medical meanings clearly.

Do NOT translate:
- Drug names or brand names
- Specific dosages or numbers
- Medical warning signs that rely on specific terms

--- English structured output ---
{content}

Respond with ONLY valid JSON matching the exact same schema:
{{
  "plain_summary": "string",
  "what_it_means_for_you": "string",
  "action_checklist": [{{"text": "string", "is_warning_sign": false}}],
  "warning_signs": ["string"],
  "low_confidence_flags": ["string"],
  "disclaimer": "string"
}}"""


def translate_structured_output(
    structured: StructuredOutput, target_language: str
) -> StructuredOutput:
    import json

    client = get_client()
    model = get_model()

    response = client.chat.completions.create(
        model=model,
        max_tokens=1500,
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
    try:
        parsed = json.loads(raw)
        return StructuredOutput(**parsed)
    except Exception:
        # Translation failed to parse -- return the original English output
        # rather than raising a 500. Surface the failure visibly.
        fallback = structured.model_copy()
        fallback.low_confidence_flags = list(structured.low_confidence_flags) + [
            f"Translation to {target_language} failed to parse. Showing original English."
        ]
        return fallback
