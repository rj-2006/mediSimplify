"""
MediSimplify API Gateway: FastAPI application defining REST endpoints
for document processing, text analysis, streaming progress, and translation.
"""
import json
import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.cross_check import cross_check
from app.models import CrossCheckResult, PipelineResponse, TextRequest, StructuredOutput
from app.ocr import prepare_image_for_ocr, run_paddle_ocr
from app.structuring import structure_document
from app.translate import translate_structured_output
from app.vision_extract import run_vision_extraction

app = FastAPI(
    title="MediSimplify API",
    description="Dual-engine AI health literacy translation platform",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}

_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_static_dir):
    app.mount("/ui", StaticFiles(directory=_static_dir, html=True), name="static")


@app.post("/process", response_model=PipelineResponse)
async def process_document(file: UploadFile = File(...)):
    """Processes an uploaded medical document (image or PDF) through the dual-engine verification pipeline."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Accepted formats: PNG, JPG, PDF.",
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        image_path = prepare_image_for_ocr(tmp_path)

        paddle_result = run_paddle_ocr(image_path)
        vision_result = run_vision_extraction(image_path)

        cc_result = cross_check(paddle_result.raw_text, vision_result.raw_text)
        structured = structure_document(cc_result)

        return PipelineResponse(ocr=cc_result, structured=structured)
    finally:
        os.remove(tmp_path)
        if image_path != tmp_path and os.path.exists(image_path):
            os.remove(image_path)


@app.post("/process-stream")
async def process_document_stream(file: UploadFile = File(...)):
    """Streams real-time Server-Sent Events (SSE) progress as each pipeline stage executes sequentially."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Accepted formats: PNG, JPG, PDF.",
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    async def event_generator():
        image_path = tmp_path
        try:
            yield f"data: {json.dumps({'step': 1, 'message': 'Step 1/4: Rasterizing document & running PaddleOCR engine...'})}\n\n"
            image_path = prepare_image_for_ocr(tmp_path)
            paddle_result = run_paddle_ocr(image_path)

            yield f"data: {json.dumps({'step': 2, 'message': 'Step 2/4: Running Vision LLM multimodal transcription...'})}\n\n"
            vision_result = run_vision_extraction(image_path)

            yield f"data: {json.dumps({'step': 3, 'message': 'Step 3/4: Reconciling extractions & verifying clinical safety...'})}\n\n"
            cc_result = cross_check(paddle_result.raw_text, vision_result.raw_text)

            yield f"data: {json.dumps({'step': 4, 'message': 'Step 4/4: Structuring plain-language clinical summary...'})}\n\n"
            structured = structure_document(cc_result)

            res = PipelineResponse(ocr=cc_result, structured=structured)
            yield f"data: {json.dumps({'step': 'done', 'result': res.model_dump()})}\n\n"
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "quota" in err_msg.lower() or "exceeded" in err_msg.lower():
                err_msg = "API Rate Limit Exceeded (Google Gemini Free Tier quota of 20 req/day reached). Please wait 1 minute or switch API keys in .env."
            yield f"data: {json.dumps({'step': 'error', 'message': err_msg})}\n\n"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if image_path != tmp_path and os.path.exists(image_path):
                os.remove(image_path)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/process-text", response_model=PipelineResponse)
async def process_pasted_text(req: TextRequest):
    """Processes direct text input by bypassing OCR and structuring directly into plain-language JSON."""
    cc_result = CrossCheckResult(
        paddle_text=req.text,
        vision_text=req.text,
        merged_text=req.text,
        disagreements=[],
        overall_confidence="high",
    )
    structured = structure_document(cc_result)
    return PipelineResponse(ocr=cc_result, structured=structured)


@app.post("/process-text-stream")
async def process_text_stream(req: TextRequest):
    """Streams real-time Server-Sent Events (SSE) progress for direct text input analysis."""
    async def event_generator():
        try:
            yield f"data: {json.dumps({'step': 1, 'message': 'Step 1/2: Preparing text & verifying clinical safety...'})}\n\n"
            cc_result = CrossCheckResult(
                paddle_text=req.text,
                vision_text=req.text,
                merged_text=req.text,
                disagreements=[],
                overall_confidence="high",
            )
            yield f"data: {json.dumps({'step': 2, 'message': 'Step 2/2: Structuring plain-language clinical summary...'})}\n\n"
            structured = structure_document(cc_result)
            res = PipelineResponse(ocr=cc_result, structured=structured)
            yield f"data: {json.dumps({'step': 'done', 'result': res.model_dump()})}\n\n"
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "quota" in err_msg.lower() or "exceeded" in err_msg.lower():
                err_msg = "API Rate Limit Exceeded (Google Gemini Free Tier quota reached). Please wait 1 minute or switch API keys in .env."
            yield f"data: {json.dumps({'step': 'error', 'message': err_msg})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/translate-structured", response_model=StructuredOutput)
async def translate_structured(structured: StructuredOutput, target_language: str):
    """Translates an existing structured health summary into the target language."""
    return translate_structured_output(structured, target_language)


@app.get("/health")
async def health():
    """Service health check endpoint."""
    return {"status": "ok", "service": "medisimplify-api", "version": "2.0.0"}
