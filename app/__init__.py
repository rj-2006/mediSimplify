# MediSimplify app package
import os
from dotenv import load_dotenv
from openai import OpenAI


def get_client() -> OpenAI:
    load_dotenv(override=True)
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    return OpenAI(api_key=api_key, base_url=base_url)


def get_model() -> str:
    load_dotenv(override=True)
    return os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"


def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


