"""
MediSimplify backend.

Pipeline per request:
  upload -> PaddleOCR + Vision LLM (parallel-ish) -> cross_check ->
  structuring -> (optional) translate -> JSON response

Run with:
    uvicorn app.main:app --reload --port 8000
"""
import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.cross_check import cross_check
from app.models import CrossCheckResult, PipelineResponse, TextRequest, StructuredOutput
from app.ocr import prepare_image_for_ocr, run_paddle_ocr
from app.structuring import structure_document
from app.translate import translate_structured_output
from app.vision_extract import run_vision_extraction

app = FastAPI(title="MediSimplify")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # weekend-project scope; lock down before real deploy
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}

# Serve the minimal frontend at /ui for quick local testing.
_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_static_dir):
    app.mount("/ui", StaticFiles(directory=_static_dir, html=True), name="static")


@app.post("/process", response_model=PipelineResponse)
async def process_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Use png/jpg/pdf.",
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        image_path = prepare_image_for_ocr(tmp_path)

        # Both extractions run against the same image. For a weekend
        # build these run sequentially; if you want real parallelism,
        # wrap each in asyncio.to_thread and await both together.
        paddle_result = run_paddle_ocr(image_path)
        vision_result = run_vision_extraction(image_path)

        cc_result = cross_check(paddle_result.raw_text, vision_result.raw_text)
        structured = structure_document(cc_result)

        return PipelineResponse(ocr=cc_result, structured=structured)
    finally:
        os.remove(tmp_path)
        if image_path != tmp_path and os.path.exists(image_path):
            os.remove(image_path)


@app.post("/process-text", response_model=PipelineResponse)
async def process_pasted_text(req: TextRequest):
    """For users who just paste text instead of uploading an image --
    skips OCR entirely, since there's nothing to extract."""
    cc_result = CrossCheckResult(
        paddle_text=req.text,
        vision_text=req.text,
        merged_text=req.text,
        disagreements=[],
        overall_confidence="high",
    )
    structured = structure_document(cc_result)
    return PipelineResponse(ocr=cc_result, structured=structured)



@app.post("/translate-structured", response_model=StructuredOutput)
async def translate_structured(structured: StructuredOutput, target_language: str):
    return translate_structured_output(structured, target_language)


@app.get("/health")
async def health():
    return {"status": "ok"}
