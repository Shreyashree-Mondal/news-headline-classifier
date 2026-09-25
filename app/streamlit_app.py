"""Streamlit web app for the News Headline Classifier.

Loads the saved LSTM and runs predictions in-process, reusing the exact
preprocessing code from the FastAPI service (api/preprocess.py).

Run locally (from the repo root):  streamlit run app/streamlit_app.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from api.preprocess import TextPipeline, clean_text  # noqa: E402

MODEL_DIR = ROOT / "models"
GITHUB_URL = "https://github.com/Shreyashree-Mondal/news-headline-classifier"

st.set_page_config(page_title="Headline topic classifier", page_icon="📰", layout="centered")

EXAMPLES = [
    "Senate passes budget bill after late-night vote",
    "Tech stocks slide as investors weigh another rate hike",
    "Underdog team clinches championship in overtime thriller",
    "Team owner lobbies Congress over stadium tax break",
]


@st.cache_resource(show_spinner="Loading the model…")
def load_model():
    import keras

    model = keras.models.load_model(MODEL_DIR / "news_lstm.keras", compile=False)
    pipeline = TextPipeline.from_dir(MODEL_DIR)
    meta_path = MODEL_DIR / "model_metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, pipeline, metadata


def classify(headline: str) -> dict:
    model, pipeline, _ = load_model()
    if not clean_text(headline):
        raise ValueError("That headline has no letters or numbers left after cleaning. Enter real words.")
    probs = np.asarray(model(pipeline.transform([headline]), training=False))[0]
    words = pipeline.words(clean_text(headline))
    ignored = pipeline.ignored_words(headline)
    k = int(probs.argmax())
    return {
        "category": pipeline.labels[k],
        "confidence": float(probs[k]),
        "probabilities": {lbl: float(p) for lbl, p in zip(pipeline.labels, probs)},
        "ignored_words": ignored,
        "word_coverage": 1 - len([w for w in words if w in ignored]) / len(words),
    }


# ---------- Page ----------
st.title("Headline topic classifier")
st.write(
    "Type a news headline and the model predicts whether it's about **politics**, "
    "**business** or **sports**. It's an LSTM trained on 15,000 HuffPost headlines "
    "with pretrained Word2Vec embeddings."
)

if "headline" not in st.session_state:
    st.session_state.headline = ""

st.caption("Try an example")
cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 2].button(ex, key=f"ex{i}", width="stretch"):
        st.session_state.headline = ex

headline = st.text_area("Headline", key="headline", max_chars=300, height=90,
                        placeholder="e.g. Central bank holds interest rates steady")

if st.button("Classify headline", type="primary", disabled=not headline.strip()):
    try:
        res = classify(headline.strip())
    except ValueError as e:
        st.error(str(e))
    else:
        st.subheader(res["category"].title())
        st.write(f"Confidence: **{res['confidence']:.1%}**")

        probs = pd.DataFrame(
            {"Probability": list(res["probabilities"].values())},
            index=[k.title() for k in res["probabilities"]],
        )
        st.bar_chart(probs, horizontal=True, height=180)

        if res["ignored_words"]:
            st.info(
                "The model couldn't use these words, because they weren't in its training "
                "vocabulary or have no Word2Vec vector: "
                + ", ".join(f"`{w}`" for w in res["ignored_words"])
                + f". It based its answer on {res['word_coverage']:.0%} of the words."
            )
        if res["confidence"] < 0.6:
            st.warning("Low confidence. This headline may mix topics, or fall outside all three.")

with st.expander("About the model"):
    _, _, info = load_model()
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{info.get('test_accuracy', 0):.1%}")
    c2.metric("Macro F1", f"{info.get('macro_f1', 0):.3f}")
    c3.metric("Test headlines", f"{info.get('test_size', 0):,}")
    if info.get("per_class"):
        st.dataframe(pd.DataFrame(info["per_class"]).T.rename(index=str.title), width="stretch")
    st.write(
        "Compared against a feed-forward network (87.0% accuracy) and a SimpleRNN (86.1%), "
        "the LSTM performed best on every class. Embeddings are frozen, so words without a "
        "Word2Vec vector contribute nothing to the prediction."
    )
    st.write(
        f"The same model is also packaged as a FastAPI REST service in a Docker container, "
        f"tested automatically on every push. [Code and results on GitHub]({GITHUB_URL})"
    )
