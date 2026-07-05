"""
Shared Pydantic data models for the MediSimplify processing pipeline.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class OCRResult(BaseModel):
    """Result from an individual text extraction engine."""
    source: str  # "paddleocr" or "vision_llm"
    raw_text: str
    confidence: Optional[float] = None


class TokenDisagreement(BaseModel):
    """Represents a medically significant discrepancy between OCR engines."""
    field_type: str  # "drug_name" | "dosage" | "frequency" | "number" | "other"
    paddle_value: str
    vision_value: str


class CrossCheckResult(BaseModel):
    """Aggregated result of cross-checking PaddleOCR and Vision LLM extractions."""
    paddle_text: str
    vision_text: str
    merged_text: str
    disagreements: List[TokenDisagreement] = Field(default_factory=list)
    overall_confidence: str  # "high" | "medium" | "low"


class ActionItem(BaseModel):
    """An actionable step for the patient."""
    text: str
    is_warning_sign: bool = False


class KeyFinding(BaseModel):
    """Structured representation of an individual lab test or prescribed medication."""
    item_name: str  # e.g., "Hemoglobin" or "Amoxicillin 500mg"
    value_or_dose: str  # e.g., "10.2 g/dL" or "1 tablet BID"
    reference_or_purpose: str  # e.g., "Normal: 12.0-17.5 g/dL" or "Bacterial infection"
    status: str  # "normal" | "attention" | "urgent" | "unconfirmed"
    explanation: str  # Plain-language explanation for this specific item


class StructuredOutput(BaseModel):
    """Patient-facing structured health summary and safety checklist."""
    overall_status: str = "NORMAL"  # "NORMAL" | "ATTENTION" | "URGENT" | "UNCONFIRMED"
    plain_summary: str
    what_it_means_for_you: str
    key_findings: List[KeyFinding] = Field(default_factory=list)
    action_checklist: List[ActionItem] = Field(default_factory=list)
    warning_signs: List[str] = Field(default_factory=list)
    low_confidence_flags: List[str] = Field(default_factory=list)
    disclaimer: str = (
        "This is an AI-generated simplification to help you understand your "
        "document. It is not medical advice. Always confirm doses and "
        "instructions with your pharmacist or doctor before acting on them."
    )


class TextRequest(BaseModel):
    """Request payload for direct text processing (/process-text)."""
    text: str


class PipelineResponse(BaseModel):
    """Complete API response containing both OCR verification and structured output."""
    ocr: CrossCheckResult
    structured: StructuredOutput
