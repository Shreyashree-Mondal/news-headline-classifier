"""FastAPI service for the news headline classifier (LSTM + frozen Word2Vec).

Run locally:   uvicorn api.main:app --reload
Docs:          http://localhost:8000/docs
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.preprocess import TextPipeline, clean_text

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
MODEL_FILE = MODEL_DIR / "news_lstm.keras"
MAX_BATCH = 64

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    import keras  # imported here so the app module loads fast in tests

    state["model"] = keras.models.load_model(MODEL_FILE, compile=False)
    state["pipeline"] = TextPipeline.from_dir(MODEL_DIR)
    meta_path = MODEL_DIR / "model_metadata.json"
    state["metadata"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    yield
    state.clear()


app = FastAPI(
    title="News Headline Classifier API",
    description=(
        "Classifies a news headline as POLITICS, BUSINESS or SPORTS using a 2-layer LSTM "
        "on frozen Word2Vec (Google News 300d) embeddings, trained on 15,000 HuffPost headlines."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- Schemas ----------
class HeadlineIn(BaseModel):
    headline: str = Field(..., min_length=1, max_length=300,
                          examples=["Senate passes budget bill after late-night vote"])


class BatchIn(BaseModel):
    headlines: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    headline: str
    category: str
    confidence: float
    probabilities: dict[str, float]
    ignored_words: list[str] = Field(
        description="Words with no usable embedding (unseen in training or missing from Word2Vec)."
    )
    word_coverage: float = Field(description="Share of words the model could actually use (0-1).")


# ---------- Helpers ----------
def _predict(headlines: list[str]) -> list[Prediction]:
    pipeline: TextPipeline = state["pipeline"]
    for h in headlines:
        if not clean_text(h):
            raise HTTPException(
                status_code=422,
                detail=f"Headline {h!r} has no letters or numbers left after cleaning. Send real words.",
            )
    X = pipeline.transform(headlines)
    probs = np.asarray(state["model"](X, training=False))
    results = []
    for h, p in zip(headlines, probs):
        words = pipeline.words(clean_text(h))
        ignored = pipeline.ignored_words(h)
        coverage = 1 - len([w for w in words if w in ignored]) / len(words)
        k = int(p.argmax())
        results.append(Prediction(
            headline=h,
            category=pipeline.labels[k],
            confidence=round(float(p[k]), 4),
            probabilities={lbl: round(float(v), 4) for lbl, v in zip(pipeline.labels, p)},
            ignored_words=ignored,
            word_coverage=round(coverage, 3),
        ))
    return results


# ---------- Routes ----------
@app.get("/", include_in_schema=False)
def root():
    return {"service": "news-headline-classifier", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    return {"labels": state["pipeline"].labels, "max_len": state["pipeline"].max_len, **state["metadata"]}


@app.post("/predict", response_model=Prediction)
def predict(body: HeadlineIn):
    return _predict([body.headline])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(body: BatchIn):
    return _predict(body.headlines)
