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

# Use 2.5-flash; if not enabled in your region, fall back to 1.5-flash
MODEL_CANDIDATES = ["gemini-2.5-flash", "gemini-1.5-flash"]
def get_model():
    for name in MODEL_CANDIDATES:
        try:
            return genai.GenerativeModel(name)
        except Exception as e:
            log.warning("Model %s not available: %s", name, e)
    raise RuntimeError("No Gemini model available (2.5-flash / 1.5-flash)")

model = get_model()

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

def _normalize_payload(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except Exception as e:
        log.warning("JSON parse failed; raw: %r err: %s", raw[:300], e)
        return {"summary": "", "action_items": [], "blockers": []}

    out = {
        "summary": data.get("summary") or "",
        "action_items": data.get("action_items") or [],
        "blockers": data.get("blockers") or [],
    }
    # force list[str]
    out["action_items"] = [str(x) for x in out["action_items"] if str(x).strip()]
    out["blockers"] = [str(x) for x in out["blockers"] if str(x).strip()]
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": getattr(model, "model_name", "unknown")}

@app.post("/analyze")
def analyze(r: Req):
    try:
        # Compose content
        content = PROMPT + (r.text or "")
        resp = model.generate_content(content, generation_config=GENCFG)

        # For visibility in Render logs:
        log.info("Candidates: %s", getattr(resp, "candidates", None))
        log.info("Text len: %s", len(resp.text or ""))

        # Use the SDK's .text (should be pure JSON because of response_mime_type)
        raw = resp.text or ""
        return _normalize_payload(raw)

    except Exception as e:
        log.error("Analyzer error: %s", e, exc_info=True)
        return {"summary": "", "action_items": [], "blockers": []}

@app.get("/selftest")
def selftest():
    sample = "Priya: finalize deck by Friday.\nSam: set up UAT by Thursday.\nBlocker: waiting on SSO approval (high)."
    resp = model.generate_content(PROMPT + sample, generation_config=GENCFG)
    return _normalize_payload(resp.text or "")
