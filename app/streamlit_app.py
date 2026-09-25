"""Streamlit front end for the News Headline Classifier API.

Set the API address in .streamlit/secrets.toml (or Streamlit Cloud secrets):
    API_URL = "https://<your-space>.hf.space"
Falls back to the API_URL environment variable, then http://localhost:8000.
"""
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Headline topic classifier", page_icon="📰", layout="centered")


def get_api_url() -> str:
    try:
        url = st.secrets.get("API_URL")
    except Exception:  # no secrets file locally
        url = None
    return (url or os.getenv("API_URL") or "http://localhost:8000").rstrip("/")


API_URL = get_api_url()

EXAMPLES = [
    "Senate passes budget bill after late-night vote",
    "Tech stocks slide as investors weigh another rate hike",
    "Underdog team clinches championship in overtime thriller",
    "Team owner lobbies Congress over stadium tax break",
]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_model_info():
    r = requests.get(f"{API_URL}/model-info", timeout=60)
    r.raise_for_status()
    return r.json()


def classify(headline: str) -> dict:
    r = requests.post(f"{API_URL}/predict", json={"headline": headline}, timeout=60)
    if r.status_code == 422:
        detail = r.json().get("detail")
        raise ValueError(detail if isinstance(detail, str) else "Enter a headline with real words.")
    r.raise_for_status()
    return r.json()


# ---------- Page ----------
st.title("Headline topic classifier")
st.write(
    "Type a news headline and the model predicts whether it's about **politics**, "
    "**business** or **sports**. It's an LSTM trained on 15,000 HuffPost headlines "
    "with pretrained Word2Vec embeddings, served through a FastAPI backend."
)

if "headline" not in st.session_state:
    st.session_state.headline = ""

st.caption("Try an example")
cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
        st.session_state.headline = ex

headline = st.text_area("Headline", key="headline", max_chars=300, height=90,
                        placeholder="e.g. Central bank holds interest rates steady")

if st.button("Classify headline", type="primary", disabled=not headline.strip()):
    try:
        with st.spinner("Classifying… the first request can take up to a minute while the API wakes up."):
            res = classify(headline.strip())
    except ValueError as e:
        st.error(str(e))
    except requests.exceptions.RequestException:
        st.error(f"Couldn't reach the API at {API_URL}. Check that it's running, then try again.")
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
    try:
        info = fetch_model_info()
    except requests.exceptions.RequestException:
        st.write("Model details are unavailable while the API is unreachable.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Test accuracy", f"{info.get('test_accuracy', 0):.1%}")
        c2.metric("Macro F1", f"{info.get('macro_f1', 0):.3f}")
        c3.metric("Test headlines", f"{info.get('test_size', 0):,}")
        if info.get("per_class"):
            st.dataframe(pd.DataFrame(info["per_class"]).T.rename(index=str.title), use_container_width=True)
        st.write(
            "Compared against a feed-forward network (87.0% accuracy) and a SimpleRNN (86.1%), "
            "the LSTM performed best on every class. Embeddings are frozen, so words without a "
            "Word2Vec vector contribute nothing to the prediction."
        )
    st.write(f"[API docs]({API_URL}/docs)")
