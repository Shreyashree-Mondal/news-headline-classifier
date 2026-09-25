# News Headline Classifier

Classifies news headlines as **Politics**, **Business** or **Sports** using neural networks on pretrained Word2Vec embeddings. The best model (a 2-layer LSTM) is packaged as a containerized **FastAPI** REST service, tested in CI on every push, and powers a live **Streamlit** web app.

- **Live demo:** `<Streamlit app URL>`

## Results

Three architectures were trained on 15,000 HuffPost headlines (5,000 per class) and evaluated on a stratified, held-out test set of 3,000 headlines.

| Model | Test accuracy | Macro F1 | Politics F1 | Business F1 | Sports F1 |
|---|---|---|---|---|---|
| Feed-forward (avg-pooled embeddings) | 86.97% | 0.870 | 0.863 | 0.862 | 0.884 |
| SimpleRNN + dense head | 86.13% | 0.861 | 0.848 | 0.864 | 0.872 |
| **2-layer LSTM + dense head** | **88.93%** | **0.889** | **0.879** | **0.885** | **0.904** |

The LSTM was the best model on every class and is the one deployed.

## How it works

```
                         ┌─► FastAPI REST service (Docker) ─► JSON: category + probabilities
Headline ─► clean → tokenize → pad ─► LSTM
                         └─► Streamlit web app (live demo)  ─► category, confidence, chart
```

Both the API and the web app share the same preprocessing module (`api/preprocess.py`) and the same saved model, so they always give identical predictions.

**Data.** News Category Dataset (Kaggle, HuffPost). The first 5,000 headlines from each of Politics, Business and Sports; headlines only (no short descriptions).

**Preprocessing.** HTML stripped, lowercased, punctuation removed except apostrophes. Keras tokenizer fitted on training data only, sequences zero-padded to the longest headline (30 tokens) so nothing is truncated.

**Split.** Seed 42; 20% of each class held out for testing, then 10% of the remainder for validation, giving 10,800 train, 1,200 validation and 3,000 test headlines.

**Embeddings.** Google News Word2Vec (300-d), loaded as frozen weights in the Embedding layer.

**Training.** Adam (lr 5e-4), batch size 256, up to 50 epochs with early stopping on validation loss; L2 regularization and dropout in the dense layers.

## What limits performance

**Vocabulary coverage.** 4,279 of the 16,193 training-vocabulary words have no Word2Vec vector. Because the embeddings are frozen, those words enter the model as zero vectors and carry no information. Lowercasing contributes to this, since Google's Word2Vec is case-sensitive (for example, names like "Biden" lose their vector once lowercased). The web app shows which words in your headline the model could not use.

**Padding direction.** Sequences are post-padded, so the SimpleRNN reads through trailing zeros before producing its final state, which weakens its memory of the actual words. This likely contributes to it trailing the feed-forward model. The LSTM's gating handles this much better.

**Recency of the sample.** Taking the first 5,000 records per category selects the most recent headlines in the dataset, so the model reflects the news topics of that period.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service and model status |
| GET | `/model-info` | Labels, test metrics, library versions |
| POST | `/predict` | `{"headline": "..."}` → category, confidence, probabilities, ignored words |
| POST | `/predict/batch` | `{"headlines": [...]}` (up to 64) |

Run the API locally (see below), then:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"headline": "Senate passes budget bill after late-night vote"}'
```

## Training/serving parity

The API does not use the Keras `Tokenizer` class at serving time. It rebuilds tokenization from an exported JSON config. To guarantee this matches training, the notebook exports 50 "golden" test headlines with their exact padded sequences and predicted probabilities, and the test suite checks that the API reproduces both. CI runs these tests on every push, then builds the Docker image and smoke-tests the running container.

## Project structure

```
notebooks/        training notebook + artifact export cell
models/           saved LSTM, preprocessing config, metadata, golden samples
api/              FastAPI service (main.py) and preprocessing (preprocess.py)
app/              Streamlit web app
tests/            pytest suite (preprocessing parity + API)
Dockerfile        API container (port 7860)
.github/workflows CI: tests, Docker build, container smoke test
```

## Run locally

```bash
# API
pip install -r requirements-dev.txt
uvicorn api.main:app --reload          # http://localhost:8000/docs
pytest -v

# Web app
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py     # loads the model directly

# Or run the API in Docker
docker build -t news-headline-classifier .
docker run -p 7860:7860 news-headline-classifier
```

## Tech stack

Python, TensorFlow/Keras, gensim (Word2Vec), scikit-learn, FastAPI, Pydantic, Docker, Streamlit, pytest, GitHub Actions, Streamlit Community Cloud.
