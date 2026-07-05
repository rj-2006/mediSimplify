# MediSimplify

> AI health-literacy translator. Upload a photo or PDF of a prescription, lab report, or discharge summary → get a plain-language summary, action checklist, and warning signs — read aloud and translatable into 5 languages.

> **Not a medical device. Not medical advice.** This is a literacy-translation prototype. It explains documents in plain language; it does not diagnose, recommend dosage changes, or replace a pharmacist or doctor.

---

## How it works

```
Upload (PNG / JPG / PDF)
    │
    ▼
prepare_image_for_ocr()        ← rasterizes PDFs; stitches multi-page into one image
    │
    ├──► PaddleOCR              ← fast, deterministic; strong on printed text & tables
    │
    └──► Vision LLM             ← robust on handwriting, stamps, messy layouts (via OpenAI SDK)
    │
    ▼
cross_check()                  ← diffs both transcriptions; flags drug name / dosage disagreements
    │
    ▼
structure_document()           ← structured LLM prompt → plain-language JSON
    │
    ▼
(optional) translate()         ← translates the already-safety-checked output
    │
    ▼
JSON response → browser renders + Web Speech API TTS
```

The dual-extraction cross-check is the core safety mechanism. Two independent methods reading the same image and disagreeing on a number is a far more reliable signal than asking one model "are you sure?"

---

## Project structure

```
MediSimplify/
├── app/
│   ├── __init__.py
│   ├── main.py           ← FastAPI app, all route definitions
│   ├── models.py         ← Pydantic data models (shared across all modules)
│   ├── ocr.py            ← PaddleOCR extraction + PDF rasterization
│   ├── vision_extract.py ← Vision LLM extraction (OpenAI SDK)
│   ├── cross_check.py    ← Reconciliation, disagreement flagging, confidence scoring
│   ├── structuring.py    ← Plain-language structuring prompt
│   └── translate.py      ← Optional translation pass
├── static/
│   └── index.html        ← Minimal single-page frontend (served at /ui)
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites
- Python **3.11** (PaddleOCR does not have wheels for 3.12+)
- An OpenAI-compatible API key (OpenAI, OpenRouter, DeepSeek, OneAPI, etc.)
- `poppler` on PATH for PDF support ([Windows binaries](https://github.com/oschwartz10612/poppler-windows/releases))

### Install

```powershell
# 1. Create a Python 3.11 venv
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key in .env file (or environment variable)
# See .env file for configuring OPENAI_API_KEY, OPENAI_BASE_URL, and MODEL_NAME
```

### Run

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

| URL | What's there |
|---|---|
| http://localhost:8000/ui | Frontend (upload, results, read-aloud, translate) |
| http://localhost:8000/docs | Interactive API docs (Swagger UI) |
| http://localhost:8000/health | Health check |

---

## API reference

### `POST /process`
Upload an image or PDF for full pipeline processing.

- **Request:** `multipart/form-data`, field `file` (`.png` / `.jpg` / `.jpeg` / `.pdf`)
- **Response:** `PipelineResponse`

```json
{
  "ocr": {
    "paddle_text": "...",
    "vision_text": "...",
    "merged_text": "...",
    "disagreements": [
      { "field_type": "dosage", "paddle_value": "500mg", "vision_value": "50mg" }
    ],
    "overall_confidence": "high | medium | low"
  },
  "structured": {
    "plain_summary": "...",
    "what_it_means_for_you": "...",
    "action_checklist": [{ "text": "...", "is_warning_sign": false }],
    "warning_signs": ["..."],
    "low_confidence_flags": ["..."],
    "disclaimer": "..."
  }
}
```

**Confidence levels:**
- `high` — both methods agree on all drug names, doses, and numbers
- `medium` — one medically-meaningful disagreement found
- `low` — two or more disagreements, or reconciliation failed

---

### `POST /process-text`
Skip OCR entirely — paste already-readable text. Useful for digital PDFs or demo fallback.

- **Request body (JSON):** `{ "text": "..." }`
- **Response:** same `PipelineResponse` shape; confidence is always `high` since there's nothing to misread

---

### `POST /translate-structured?target_language=Hindi`
Translate an existing structured result into another language. Runs *after* the safety-checked English output, so scope rules only need to be enforced once.

- **Query param:** `target_language` — any language name (e.g. `Hindi`, `Spanish`, `Bengali`, `Tamil`, `French`)
- **Request body:** `StructuredOutput` JSON (the `structured` field from a previous `/process` response)
- **Response:** `StructuredOutput` in the target language; drug names are left untranslated

---

### `GET /health`
Returns `{ "status": "ok" }`. Use for uptime checks.

---

## Data models

### `OCRResult`
| Field | Type | Notes |
|---|---|---|
| `source` | `str` | `"paddleocr"` or `"vision_llm"` |
| `raw_text` | `str` | Raw transcribed text |
| `confidence` | `float \| None` | Per-line average from PaddleOCR; `None` for vision LLM |

### `CrossCheckResult`
| Field | Type | Notes |
|---|---|---|
| `paddle_text` | `str` | Raw PaddleOCR output |
| `vision_text` | `str` | Raw Vision LLM output |
| `merged_text` | `str` | Best-guess reconciled transcription |
| `disagreements` | `list[TokenDisagreement]` | Medically-relevant mismatches only |
| `overall_confidence` | `str` | `high` / `medium` / `low` |

### `StructuredOutput`
| Field | Type | Notes |
|---|---|---|
| `plain_summary` | `str` | 2–4 sentences: what the document is and says |
| `what_it_means_for_you` | `str` | 2–4 sentences: practical meaning for the patient |
| `action_checklist` | `list[ActionItem]` | Concrete steps; `is_warning_sign` flags ER-level items |
| `warning_signs` | `list[str]` | Symptoms warranting a doctor/ER visit |
| `low_confidence_flags` | `list[str]` | Items the user must manually verify |
| `disclaimer` | `str` | Hardcoded — always present, cannot be suppressed by the prompt |

---

## The one rule that matters most

The structuring prompt (`app/structuring.py`) is written to **never** suggest dosage changes or tell the user to start, stop, or skip a medication — only to explain what's already written, and to flag uncertainty instead of smoothing over it.

If you modify that prompt, keep this rule intact. It's the difference between a useful accessibility tool and something that could cause real harm if it's confidently wrong.

---

## Known limitations & future work

| Item | Status |
|---|---|
| OCR + Vision run sequentially | Wrap in `asyncio.gather` to halve latency |
| No auth / rate limiting / file size cap | Fine for demo; required before public deploy |
| No persistence | By design for MVP; add a DB layer if you extend this |
| PDF support requires `poppler` on PATH | See setup instructions |
| PaddleOCR requires Python ≤ 3.11 | Use the provided venv setup |
| Camera capture on mobile | Not built; `<input capture="environment">` is a one-line addition |
| Drug interaction checking | Roadmap only — do not build without a verified medical database |
