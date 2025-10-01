import os, json, logging
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gemini-proxy")

API_KEY = os.getenv("GEMINI_API_KEY", "")
if not API_KEY:
    log.error("GEMINI_API_KEY is not set!")
genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash"
try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    log.warning("Falling back to 1.5-flash: %s", e)
    MODEL_NAME = "gemini-1.5-flash"
    model = genai.GenerativeModel(MODEL_NAME)

app = FastAPI()

class Req(BaseModel):
    text: str

PROMPT = (
    "You are a meeting analyzer. Read the transcript and return STRICT JSON ONLY with this schema:\n"
    "{\n"
    '  "summary": "2-4 sentences summarizing outcomes and decisions",\n'
    '  "action_items": ["Owner: task by date", "..."],\n'
    '  "blockers": ["Team/Owner: blocker (severity)"]\n'
    "}\n"
    "No markdown, no prose, only JSON. Transcript follows:\n"
)

GENCFG = GenerationConfig(
    temperature=0.2,
    response_mime_type="application/json",
)

def normalize(raw: str) -> dict:
    try:
        data = json.loads(raw or "")
    except Exception as e:
        log.warning("JSON parse failed; raw head=%r err=%s", (raw or "")[:200], e)
        return {"summary": "", "action_items": [], "blockers": []}
    return {
        "summary": data.get("summary") or "",
        "action_items": [str(x) for x in (data.get("action_items") or []) if str(x).strip()],
        "blockers": [str(x) for x in (data.get("blockers") or []) if str(x).strip()],
    }

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}

@app.get("/selftest")
def selftest():
    sample = "Priya: finalize deck by Friday.\nSam: set up UAT by Thursday.\nBlocker: waiting on SSO approval (high)."
    resp = model.generate_content(PROMPT + sample, generation_config=GENCFG)
    return normalize(resp.text or "")

@app.post("/analyze")
def analyze(r: Req):
    content = PROMPT + (r.text or "")
    resp = model.generate_content(content, generation_config=GENCFG)
    return normalize(resp.text or "")

# Debug: see raw model output if needed
@app.post("/debug_analyze")
def debug_analyze(r: Req):
    content = PROMPT + (r.text or "")
    resp = model.generate_content(content, generation_config=GENCFG)
    return {"raw": resp.text or ""}
