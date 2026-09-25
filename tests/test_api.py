import numpy as np


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["labels"] == ["POLITICS", "BUSINESS", "SPORTS"]


def test_predict_shape(client):
    r = client.post("/predict", json={"headline": "Senate passes budget bill after late-night vote"})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] in body["probabilities"]
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert 0 <= body["word_coverage"] <= 1


def test_rejects_empty_and_symbol_only(client):
    assert client.post("/predict", json={"headline": ""}).status_code == 422
    assert client.post("/predict", json={"headline": "?!?!"}).status_code == 422


def test_batch(client):
    heads = ["Stocks fall on rate fears", "Team wins title in overtime"]
    r = client.post("/predict/batch", json={"headlines": heads})
    assert r.status_code == 200
    assert [p["headline"] for p in r.json()] == heads


def test_predictions_match_notebook(client, golden):
    """The served model must reproduce the notebook's probabilities."""
    r = client.post("/predict/batch", json={"headlines": [g["text"] for g in golden]})
    assert r.status_code == 200
    labels = ["POLITICS", "BUSINESS", "SPORTS"]
    for pred, g in zip(r.json(), golden):
        served = [pred["probabilities"][l] for l in labels]
        np.testing.assert_allclose(served, g["probabilities"], atol=1e-3)
