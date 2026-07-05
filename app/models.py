"""
Shared data models for the MediSimplify pipeline.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class OCRResult(BaseModel):
    source: str  # "paddleocr" or "vision_llm"
    raw_text: str
    confidence: Optional[float] = None  # PaddleOCR gives per-line conf; vision LLM won't


class TokenDisagreement(BaseModel):
    """A specific point where the two extractions disagree on something
    that matters (drug name, dose, frequency, number)."""
    field_type: str  # "drug_name" | "dosage" | "frequency" | "number" | "other"
    paddle_value: str
    vision_value: str


class CrossCheckResult(BaseModel):
    paddle_text: str
    vision_text: str
    merged_text: str
    disagreements: List[TokenDisagreement] = Field(default_factory=list)
    overall_confidence: str  # "high" | "medium" | "low"


class ActionItem(BaseModel):
    text: str
    is_warning_sign: bool = False


class StructuredOutput(BaseModel):
    plain_summary: str
    what_it_means_for_you: str
    action_checklist: List[ActionItem]
    warning_signs: List[str]
    low_confidence_flags: List[str] = Field(default_factory=list)
    disclaimer: str = (
        "This is an AI-generated simplification to help you understand your "
        "document. It is not medical advice. Always confirm doses and "
        "instructions with your pharmacist or doctor before acting on them."
    )


class TextRequest(BaseModel):
    """Body for /process-text — accepts pasted document text."""
    text: str


class PipelineResponse(BaseModel):
    ocr: CrossCheckResult
    structured: StructuredOutput
