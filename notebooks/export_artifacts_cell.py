# =====================================================================
# EXPORT ARTIFACTS FOR DEPLOYMENT
# Paste this as the LAST cell of Mondal_HW2.ipynb and run it after the
# notebook has run top to bottom (it uses lstm_model, tokenizer, MAX_LEN,
# missing, target_categories, X_test_pad, y_test, X_test from earlier cells).
# It writes everything the API needs into a "models/" folder.
# =====================================================================
import os, json, sys, platform
import numpy as np
import tensorflow as tf
import keras
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

EXPORT_DIR = "models"
os.makedirs(EXPORT_DIR, exist_ok=True)

# 1) The trained LSTM (best model: ~90% test accuracy)
lstm_model.save(os.path.join(EXPORT_DIR, "news_lstm.keras"))

# 2) Preprocessing config as plain JSON.
#    The API re-implements the tokenizer from this file, so serving does not
#    depend on the (deprecated) Keras Tokenizer class.
preprocessing = {
    "word_index": tokenizer.word_index,
    "num_words": tokenizer.num_words,
    "oov_token": tokenizer.oov_token,
    "filters": tokenizer.filters,
    "lower": tokenizer.lower,
    "split": tokenizer.split,
    "max_len": int(MAX_LEN),
    "padding": "post",
    "truncating": "post",
    "labels": list(target_categories),  # position = class index
}
with open(os.path.join(EXPORT_DIR, "preprocessing.json"), "w") as f:
    json.dump(preprocessing, f)

# 3) Words that are in the vocabulary but have no Word2Vec vector
#    (they were zero rows in the frozen embedding matrix).
with open(os.path.join(EXPORT_DIR, "w2v_missing_words.json"), "w") as f:
    json.dump(sorted(set(missing)), f)

# 4) Test-set metrics + library versions (shown by the API's /model-info)
probs = lstm_model.predict(X_test_pad, verbose=0)
y_pred = probs.argmax(axis=1)
p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2], zero_division=0)
micro = precision_recall_fscore_support(y_test, y_pred, average="micro")
macro = precision_recall_fscore_support(y_test, y_pred, average="macro")
metadata = {
    "model_name": "LSTM (2-layer) + dense head, frozen Word2Vec (Google News 300d)",
    "test_size": int(len(y_test)),
    "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    "micro_f1": round(float(micro[2]), 4),
    "macro_f1": round(float(macro[2]), 4),
    "per_class": {
        target_categories[i]: {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f1[i]), 4),
        }
        for i in range(len(target_categories))
    },
    "vocab_size": len(tokenizer.word_index) + 1,
    "w2v_missing_count": len(set(missing)),
    "versions": {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
    },
}
with open(os.path.join(EXPORT_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# 5) "Golden" samples: 50 test headlines with the exact sequences and
#    probabilities produced HERE. The repo's tests check that the API
#    reproduces them, which catches any training/serving mismatch.
rng = np.random.default_rng(42)
idx = rng.choice(len(X_test), size=min(50, len(X_test)), replace=False)
golden = [
    {
        "text": X_test[i],
        "sequence": X_test_pad[i].tolist(),
        "probabilities": probs[i].astype(float).tolist(),
    }
    for i in idx
]
with open(os.path.join(EXPORT_DIR, "golden_samples.json"), "w") as f:
    json.dump(golden, f)

# 6) Sanity check: reload from disk and confirm accuracy is unchanged
reloaded = keras.models.load_model(os.path.join(EXPORT_DIR, "news_lstm.keras"))
reloaded_acc = accuracy_score(y_test, reloaded.predict(X_test_pad, verbose=0).argmax(axis=1))
print(json.dumps(metadata, indent=2))
print(f"\nReloaded model test accuracy: {reloaded_acc:.4f}  (should match {metadata['test_accuracy']})")
print("Saved files:", os.listdir(EXPORT_DIR))
print("\n>>> Put these versions in api/requirements.txt:")
print(f"tensorflow-cpu=={tf.__version__}\nkeras=={keras.__version__}")
