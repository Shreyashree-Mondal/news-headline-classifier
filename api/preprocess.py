"""Text preprocessing for serving.

Mirrors the training notebook exactly:
  clean_text()  -> same cleaning function used in Mondal_HW2.ipynb
  TextPipeline  -> re-implements Keras Tokenizer.texts_to_sequences + pad_sequences
                   from the exported preprocessing.json, so serving does not need
                   the deprecated Keras Tokenizer class.
tests/test_preprocess.py checks this against sequences produced in the notebook.
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def clean_text(text: str) -> str:
    """Identical to the notebook's clean_text()."""
    text = str(text)
    text = BeautifulSoup(text, "html.parser").get_text()  # removes HTML tags
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TextPipeline:
    def __init__(self, config: dict, w2v_missing: set[str] | None = None):
        self.word_index: dict[str, int] = config["word_index"]
        self.num_words: int | None = config.get("num_words")
        self.oov_token: str | None = config.get("oov_token")
        self.filters: str = config.get("filters", "")
        self.lower: bool = config.get("lower", True)
        self.split: str = config.get("split", " ")
        self.max_len: int = int(config["max_len"])
        self.padding: str = config.get("padding", "post")
        self.truncating: str = config.get("truncating", "post")
        self.labels: list[str] = config["labels"]
        self.oov_index = self.word_index.get(self.oov_token) if self.oov_token else None
        self.w2v_missing = w2v_missing or set()

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "TextPipeline":
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "preprocessing.json").read_text())
        missing_path = model_dir / "w2v_missing_words.json"
        missing = set(json.loads(missing_path.read_text())) if missing_path.exists() else set()
        return cls(config, missing)

    # --- Keras Tokenizer behaviour -------------------------------------
    def words(self, text: str) -> list[str]:
        """Keras text_to_word_sequence()."""
        if self.lower:
            text = text.lower()
        text = text.translate(str.maketrans({c: self.split for c in self.filters}))
        return [w for w in text.split(self.split) if w]

    def to_sequence(self, text: str) -> list[int]:
        """Keras Tokenizer.texts_to_sequences() for a single text."""
        seq = []
        for w in self.words(text):
            i = self.word_index.get(w)
            if i is not None:
                if self.num_words and i >= self.num_words:
                    if self.oov_index is not None:
                        seq.append(self.oov_index)
                else:
                    seq.append(i)
            elif self.oov_index is not None:
                seq.append(self.oov_index)
        return seq

    def pad(self, seq: list[int]) -> list[int]:
        """Keras pad_sequences() for a single sequence."""
        if len(seq) > self.max_len:
            seq = seq[: self.max_len] if self.truncating == "post" else seq[-self.max_len :]
        pad = [0] * (self.max_len - len(seq))
        return seq + pad if self.padding == "post" else pad + seq

    # --- Public API -----------------------------------------------------
    def transform(self, raw_texts: list[str]) -> np.ndarray:
        return np.array([self.pad(self.to_sequence(clean_text(t))) for t in raw_texts], dtype="int32")

    def ignored_words(self, raw_text: str) -> list[str]:
        """Words that reach the model as all-zero vectors: either never seen in
        training (OOV) or seen but missing from Word2Vec."""
        out = []
        for w in self.words(clean_text(raw_text)):
            if w not in self.word_index or w in self.w2v_missing:
                if w not in out:
                    out.append(w)
        return out
