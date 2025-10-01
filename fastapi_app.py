# fastapi_app.py
# pip install fastapi uvicorn google-genai==0.3.0
import os, json
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()

class Req(BaseModel):
    text: str

PROMPT = """
You are a meeting analyzer. Read the transcript and return STRICT JSON ONLY with this schema:
{
  "summary": "2-4 sentences summarizing key outcomes and decisions",
  "action_items": ["Owner: task by date", "..."],
  "blockers": ["Team/Owner: blocker (severity)"]
}
No markdown, no prose, only JSON. Transcript follows:
"""

@app.post("/analyze")
def analyze(r: Req):
    """
    Returns only the in-scope keys. If Gemini returns anything non-JSON,
    the endpoint still responds with a valid (empty) JSON structure.
    """
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": PROMPT + r.text}]}],
            # This forces Gemini to emit application/json, not prose:
            generation_config={"response_mime_type": "application/json"}
        )

        # In google-genai, the simplest way is to use .text for the top candidate:
        raw_text = resp.text or ""
        data = json.loads(raw_text)

        # Keep ONLY the three keys we need (scope):
        out = {
            "summary": data.get("summary", "") or "",
            "action_items": data.get("action_items", []) or [],
            "blockers": data.get("blockers", []) or []
        }
        # Final type-safety: ensure list of strings
        out["action_items"] = [str(x) for x in out["action_items"]]
        out["blockers"] = [str(x) for x in out["blockers"]]
        return out

    except Exception as e:
        # Log to server console for debugging on Render:
        print("Analyzer error:", repr(e))
        return {"summary": "", "action_items": [], "blockers": []}
