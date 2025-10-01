import os, json
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()

class Req(BaseModel):
    text: str

PROMPT = """You are a meeting analyzer.
Return STRICT JSON ONLY with keys:
- summary: 2-4 sentences
- action_items: array of strings like "Owner: task by date"
- blockers: array of strings like "Team/Owner: blocker (severity)"
Transcript:
"""

@app.post("/analyze")
def analyze(r: Req):
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=PROMPT + r.text
        )
        try:
            data = json.loads(resp.text)
        except Exception:
            data = {}
        # keep only in-scope keys
        return {
            "summary": data.get("summary", ""),
            "action_items": data.get("action_items", []),
            "blockers": data.get("blockers", [])
        }
    except Exception as e:
        # never crash: return an empty-but-valid payload
        return {"summary": "(analysis fallback)", "action_items": [], "blockers": []}
