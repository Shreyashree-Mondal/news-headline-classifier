import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "models"


def _load(name):
    path = MODEL_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not found in models/ - run the notebook export cell first")
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def golden():
    return _load("golden_samples.json")


@pytest.fixture(scope="session")
def pipeline():
    _load("preprocessing.json")
    from api.preprocess import TextPipeline
    return TextPipeline.from_dir(MODEL_DIR)


@pytest.fixture(scope="session")
def client():
    if not (MODEL_DIR / "news_lstm.keras").exists():
        pytest.skip("news_lstm.keras not found in models/")
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c
