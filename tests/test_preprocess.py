from api.preprocess import clean_text


def test_clean_text_matches_notebook_rules():
    assert clean_text("<b>Biden</b> Says U.S. Forces Would Defend Taiwan!") == "biden says u s forces would defend taiwan"
    assert clean_text("Musk's   'Twitter'  Deal — Off?") == "musk's 'twitter' deal off"
    assert clean_text("   ") == ""


def test_clean_text_is_idempotent():
    s = clean_text("Crypto Crash Leaves Coinbase Reeling, 2022")
    assert clean_text(s) == s


def test_sequences_match_training_tokenizer(pipeline, golden):
    """Training/serving parity: the API must build exactly the same padded
    sequences as the Keras Tokenizer did in the notebook."""
    for sample in golden:
        assert pipeline.transform([sample["text"]])[0].tolist() == sample["sequence"], sample["text"]


def test_padding_and_truncation(pipeline):
    long_text = " ".join(["word"] * (pipeline.max_len + 10))
    seq = pipeline.transform([long_text])[0]
    assert len(seq) == pipeline.max_len
    assert len(pipeline.transform(["senate"])[0]) == pipeline.max_len
