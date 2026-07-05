# MediSimplify — Dual-Engine AI Health Literacy Platform

> **Transforming complex medical prescriptions, lab reports, and discharge summaries into clear, actionable plain English with automated clinical safety verification in under 30 seconds.**

> [!IMPORTANT]
> **Clinical Disclaimer:** MediSimplify is an AI-powered health literacy translation platform. It explains complex medical documentation in plain language (grade 6–8 reading level); it **does not** diagnose medical conditions, recommend dosage adjustments, or replace professional consultation with a physician or pharmacist. Always confirm doses and instructions with healthcare professionals.

---

## 🌟 Executive Summary & Mission

Navigating medical documentation is often daunting for patients, elderly individuals, non-native speakers, and those with low health literacy. Traditional medical reports and handwritten prescriptions are dense with clinical jargon, abbreviations, and complex numerical ranges. 

**MediSimplify** bridges this gap by providing instant clarity without panic. Unlike standard text simplifiers that rely on a single AI model—which can hallucinate critical drug doses or numerical values—MediSimplify introduces an enterprise-grade **Dual-Engine OCR Verification Engine**. By independently cross-checking deterministic optical character recognition against multimodal vision LLMs, MediSimplify ensures that every number, unit, and medication name is verified before presenting a plain-language summary to the patient.

---

## 🛡️ The Competitive Advantage: Why Dual-Engine Verification Beats Single LLMs

Most medical AI tools (such as traditional single-LLM pipelines) pass raw image text directly to an LLM and ask it to summarize. This introduces a critical clinical safety risk: if an OCR engine misreads a dosage (e.g., reading *5.0mg* as *50mg*), a single LLM will confidently generate a summary based on that erroneous number.

MediSimplify eliminates this single-point-of-failure through **Dual-Engine Cross-Checking**:

```
                  ┌────────────────────────────────────────┐
                  │       Medical Report (Image / PDF)     │
                  └───────────────────┬────────────────────┘
                                      │
                   prepare_image_for_ocr() [PDF Rasterization]
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
   ┌───────────────────────────┐             ┌───────────────────────────┐
   │     Engine 1: PaddleOCR   │             │   Engine 2: Vision LLM    │
   │  (Deterministic & Fast)   │             │  (Multimodal & Robust)    │
   └─────────────┬─────────────┘             └─────────────┬─────────────┘
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                        cross_check() [Reconciliation]
                • Extracts standardized dosage & unit tokens
                • Diffs numerical values & medication names
                • Assigns Confidence Score (HIGH / MEDIUM / LOW)
                                      │
                                      ▼
                      structure_document() [LLM Structuring]
                • Generates Clinical Risk Status (Normal/Attention/Urgent)
                • Extracts Granular Key Findings Breakdown
                                      │
                                      ▼
                   Gruvbox UI Dashboard • TTS Read-Aloud • Print
```

1. **Deterministic OCR (PaddleOCR):** Excels at structured tabular data, printed lab results, and standard typography.
2. **Multimodal Vision LLM:** Excels at deciphering messy doctor handwriting, hospital stamps, letterheads, and medical abbreviations.
3. **Reconciliation & Discrepancy Flagging:** If Engine 1 reads *"Amoxicillin 500mg"* and Engine 2 reads *"Amoxicillin 50mg"*, the cross-check engine immediately flags a **medically significant discrepancy**, downgrades confidence to `LOW`, and alerts the patient to verify the original document with their doctor.

---

## ✨ Key Features & UI Highlights

- 🎨 **Gruvbox Dark & Light Theme UI:** Designed with rich aesthetics, glassmorphic containers, and typography powered by Google Fonts (*Outfit* and *Inter*). Features an instant toggle between Gruvbox Dark (`#282828` background) and Gruvbox Light (`#fbf1c7` background).
- 🚨 **Clinical Risk Status Banners:** Instantly categorizes reports into visual safety levels:
  - `🟢 NORMAL`: Routine findings with no alarming values.
  - `🟡 ATTENTION`: Values outside reference ranges requiring physician follow-up.
  - `🔴 URGENT`: Critical lab values or acute emergency warnings requiring immediate medical attention.
- 🧪 **Structured Key Findings Breakdown:** Granular extraction of individual lab tests and prescribed medications, displaying item names, measured values/doses, reference ranges, status badges, and plain-English explanations.
- 🔊 **Web Speech TTS Read-Aloud:** Built-in audio playback allows visually impaired or elderly patients to listen to their health summary read aloud with natural speech synthesis.
- 🌐 **Instant Multilingual Translation:** Translate structured health summaries into **Hindi, Spanish, Bengali, Tamil, French, German, or Arabic** with a single click while automatically keeping medication names and numerical dosages in standard clinical format for safety.
- 🖨️ **Print & Save as PDF:** Dedicated print stylesheet (`@media print`) strips navigation bars and toolbars, formatting the results dashboard cleanly for physical printing or saving as a PDF for doctor visits.
- 📁 **Multi-Format & Multi-Page Support:** Seamlessly processes PNG, JPG, JPEG, and multi-page PDF documents (automatically stitching PDF pages into unified high-resolution images).

---

## 🛠️ Tech Stack & Architecture

| Component | Technology | Description |
|---|---|---|
| **API Gateway & Router** | **FastAPI** (Python 3.11) | High-performance asynchronous REST API with CORS and static file serving. |
| **Deterministic OCR** | **PaddleOCR 2.x** | Fast local optical character recognition engine specialized in tabular data. |
| **Vision & Structuring LLM** | **OpenAI / Google Gemini** | Powered via OpenAI SDK compatibility layer (supports Gemini 1.5 Flash, GPT-4o, OpenRouter). |
| **Data Validation** | **Pydantic v2** | Strict type-checked schemas ensuring JSON integrity across all pipeline stages. |
| **Frontend Dashboard** | **Vanilla HTML5 / CSS / JS** | Zero-dependency, responsive Single Page Application with Gruvbox color tokens. |
| **PDF Processing** | **pdf2image / Poppler** | High-DPI rasterization and vertical image stitching for multi-page medical records. |

---

## ⚙️ Dynamic Multi-Provider LLM Setup

MediSimplify is built on an OpenAI-compatible client architecture, allowing seamless integration with multiple LLM providers without code modifications. Configure your `.env` file in the project root:

```ini
# Option 1: Google Gemini (Recommended — Fast & Multimodal)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
MODEL_NAME=gemini-flash-latest

# Option 2: OpenAI Official
# OPENAI_API_KEY=your_openai_api_key_here
# MODEL_NAME=gpt-4o-mini

# Option 3: OpenRouter / DeepSeek / Custom
# OPENAI_API_KEY=your_openrouter_key_here
# OPENAI_BASE_URL=https://openrouter.ai/api/v1
# MODEL_NAME=anthropic/claude-3-5-sonnet
```

---

## 🔌 API Reference & Schema Definitions

### `POST /process`
Processes an uploaded image or PDF document through the full dual-engine verification pipeline.
- **Request:** `multipart/form-data` with field `file` (`.png`, `.jpg`, `.jpeg`, `.pdf`).
- **Response:** `PipelineResponse` (JSON)

```json
{
  "ocr": {
    "source": "vision_llm",
    "paddle_text": "Amoxicillin 500mg BID",
    "vision_text": "Amoxicillin 500mg BID",
    "merged_text": "Amoxicillin 500mg BID",
    "disagreements": [],
    "overall_confidence": "high"
  },
  "structured": {
    "overall_status": "NORMAL",
    "plain_summary": "This is a prescription for Amoxicillin, an antibiotic medication used to treat bacterial infections.",
    "what_it_means_for_you": "You should take this medication exactly as directed by your doctor until the course is finished.",
    "key_findings": [
      {
        "item_name": "Amoxicillin",
        "value_or_dose": "500mg BID (twice daily)",
        "reference_or_purpose": "To treat bacterial infection",
        "status": "normal",
        "explanation": "An antibiotic capsule taken two times every day to clear up infection."
      }
    ],
    "action_checklist": [
      {
        "text": "Take one 500mg capsule by mouth twice a day.",
        "is_warning_sign": false
      }
    ],
    "warning_signs": [
      "Severe skin rash, difficulty breathing, or swelling of the face and throat."
    ],
    "low_confidence_flags": [],
    "disclaimer": "This is an AI-generated simplification to help you understand your document. It is not medical advice. Always confirm doses and instructions with your pharmacist or doctor before acting on them."
  }
}
```

---

### `POST /process-text`
Bypasses image OCR to directly analyze pasted digital report text or EHR notes.
- **Request Body:** `{ "text": "Hemoglobin: 10.2 g/dL (Normal: 12.0-17.5 g/dL)..." }`
- **Response:** Same `PipelineResponse` schema with `overall_confidence` set to `high`.

---

### `POST /translate-structured?target_language=Hindi`
Translates an existing structured summary into the specified target language while preserving clinical numerical safety.
- **Query Parameter:** `target_language` (e.g., `Hindi`, `Spanish`, `Bengali`, `Tamil`, `French`, `German`, `Arabic`).
- **Request Body:** `StructuredOutput` JSON payload from a previous processing request.
- **Response:** Translated `StructuredOutput` JSON payload.

---

### `GET /health`
Service health check endpoint for monitoring and uptime verification.
- **Response:** `{ "status": "ok", "service": "medisimplify-api", "version": "2.0.0" }`

---

## 🚀 Local Setup & Installation

### Prerequisites
1. **Python 3.11** (Required: PaddleOCR pre-compiled wheels are optimized for Python ≤ 3.11).
2. **Poppler** (Required for PDF rasterization):
   - **Windows:** Download from [Poppler Windows Releases](https://github.com/oschwartz10612/poppler-windows/releases) and add `bin/` to your system `PATH`.
   - **macOS:** `brew install poppler`
   - **Linux:** `sudo apt-get install poppler-utils`

---

### 🐍 Virtual Environment Guide & Troubleshooting (`ModuleNotFoundError`)

> [!WARNING]
> **Why use a Virtual Environment?** If you type `python` directly in your terminal without activating your virtual environment, your operating system will execute your **global system Python** (e.g., `C:\Python314\python.exe`). Because project dependencies like `openai`, `fastapi`, and `paddleocr` are installed inside the virtual environment (`venv`), running global Python will cause errors like:
> `ModuleNotFoundError: No module named 'openai'` or `No module named 'fastapi'`

To ensure you are using the correct isolated environment, follow these instructions based on your operating system:

#### 1. Windows (PowerShell / Command Prompt)
You have two reliable ways to run the server on Windows:

- **Option A: Activate the environment first (Recommended)**
  ```powershell
  # 1. Clone the repository and navigate to the directory
  git clone https://github.com/rj-2006/mediSimplify.git
  cd MediSimplify

  # 2. Create virtual environment
  python -m venv venv

  # 3. Activate virtual environment (Look for (venv) in your prompt!)
  .\venv\Scripts\activate

  # 4. Install dependencies & configure .env
  pip install -r requirements.txt
  cp .env.example .env

  # 5. Launch development server
  python -m uvicorn app.main:app --reload --port 8000
  ```

- **Option B: Execute directly via virtual environment binary (No activation needed)**
  If your PowerShell execution policy blocks `activate.ps1` or you want a quick one-liner without activating:
  ```powershell
  .\venv\Scripts\pip.exe install -r requirements.txt
  .\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
  ```

#### 2. macOS & Linux (Bash / Zsh)
```bash
# 1. Clone repository & create virtual environment
git clone https://github.com/rj-2006/mediSimplify.git
cd MediSimplify
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install dependencies & launch server
pip install -r requirements.txt
cp .env.example .env
python -m uvicorn app.main:app --reload --port 8000
```

---

### Accessing the Platform
Once the server is running, access the platform:
- **Interactive Gruvbox UI Dashboard:** [http://localhost:8000/ui](http://localhost:8000/ui)
- **FastAPI Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)


---

## 🔒 Patient Safety & Scope Discipline

The core structuring engine (`app/structuring.py`) enforces strict clinical boundaries:
1. **No Clinical Judgment:** The AI is strictly prohibited from suggesting dosage modifications, treatment cessations, or alternative therapies.
2. **Transparent Uncertainty:** Any token discrepancy between PaddleOCR and Vision LLM is surfaced visibly to the user in the UI rather than smoothed over by language models.
3. **Emergency Warning Signs:** Symptom alerts are tailored specifically to emergency clinical signs associated with the reported medications or test findings.

---

## 📄 License & Acknowledgments

This project is licensed under the MIT License. Built with dedication to advancing accessible, safe, and transparent patient health literacy.
