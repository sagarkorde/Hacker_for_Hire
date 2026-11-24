"""REST analysis service: POST /analyze returns the HaaS decision, score and entities.

Run:  uvicorn service.app:app --port 8000   (from the repository root)
"""
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import rules  # noqa: E402

app = FastAPI(title="HaaS listing analyzer")


class Doc(BaseModel):
    text: str
    url: str | None = None


@app.post("/analyze")
def analyze(doc: Doc):
    p, s, e, sc = rules.score(doc.text)
    label, _ = rules.predict(doc.text)
    return {"url": doc.url, "haas": bool(label), "score": sc, "primary_hits": p, "secondary_hits": s,
            "entity_cue": bool(e), "entities": rules.entities(doc.text), "threshold": rules.THRESHOLD}
