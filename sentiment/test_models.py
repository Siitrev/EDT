"""Testy nakladki na modele sentymentu.

Szybkie testy uzywaja podstawionego klasyfikatora (bez pobierania modeli).
Testy oznaczone `slow` laduja prawdziwe modele z HuggingFace:
    python -m pytest sentiment/test_models.py -m slow -v
"""
import pytest

from sentiment import models as m


def test_prepare_text_keeps_case_and_replaces_links_and_users():
    assert m.prepare_text("  Brawo @Jan_Kowalski!  zobacz https://x.pl/a?b=1 ​") == "Brawo @user! zobacz http"


@pytest.mark.parametrize("text", [None, "", "   \n\t", "​﻿"])
def test_prepare_text_empty(text):
    assert m.prepare_text(text) == ""


def test_prepare_text_unescapes_html():
    assert m.prepare_text("Tak &amp; nie") == "Tak & nie"


def test_variant_text_raw_and_clean():
    t = "Brawo @Jan! https://x.pl SUPER!!!!!!"
    assert m.variant_text(t, "raw") == "Brawo @user! http SUPER!!!!!!"
    assert m.variant_text(t, "clean") == "brawo user ! url super!!!"


def test_variant_text_unknown():
    with pytest.raises(ValueError):
        m.variant_text("x", "lemma")


def test_normalize_scores_three_class_case_insensitive():
    spec = m.MODEL_SPECS["polbert_politics"]
    raw = [{"label": "Negative", "score": 0.7}, {"label": "Neutral", "score": 0.2}, {"label": "Positive", "score": 0.1}]
    assert m.normalize_scores(raw, spec.label_map) == pytest.approx({"negative": 0.7, "neutral": 0.2, "positive": 0.1})


def test_normalize_scores_five_class_merges():
    spec = m.MODEL_SPECS["distilbert_multi"]
    raw = [{"label": "Very Negative", "score": 0.1}, {"label": "Negative", "score": 0.2},
           {"label": "Neutral", "score": 0.3}, {"label": "Positive", "score": 0.25}, {"label": "Very Positive", "score": 0.15}]
    s = m.normalize_scores(raw, spec.label_map)
    assert s == pytest.approx({"negative": 0.3, "neutral": 0.3, "positive": 0.4})
    assert sum(s.values()) == pytest.approx(1.0)


def test_normalize_scores_unknown_label_raises():
    with pytest.raises(ValueError, match="Nieznana etykieta"):
        m.normalize_scores([{"label": "joy", "score": 1.0}], m.MODEL_SPECS["herbert_twitter"].label_map)


def test_top_and_binary_label():
    s = {"negative": 0.2, "neutral": 0.5, "positive": 0.3}
    assert m.top_label(s) == "neutral"
    assert m.binary_label(s) == "positive"
    assert m.binary_label({"negative": 0.4, "neutral": 0.2, "positive": 0.4}) == "positive"


def test_model_specs_registry():
    assert list(m.MODEL_SPECS) == ["herbert_twitter", "polbert_politics", "distilbert_multi"]
    assert m.MODEL_SPECS["herbert_twitter"].hf_id == "bardsai/twitter-sentiment-pl-base"
    assert m.MODEL_SPECS["polbert_politics"].hf_id == "eevvgg/bert-polish-sentiment-politics"
    assert m.MODEL_SPECS["distilbert_multi"].hf_id == "tabularisai/multilingual-sentiment-analysis"
    for spec in m.MODEL_SPECS.values():
        assert set(spec.label_map.values()) == set(m.LABELS)


def test_sentiment_model_predict_uses_classifier_in_order():
    def fake(texts):
        return [[{"label": "positive", "score": 0.9 if "dobr" in t else 0.1},
                 {"label": "neutral", "score": 0.0},
                 {"label": "negative", "score": 0.1 if "dobr" in t else 0.9}] for t in texts]
    model = m.SentimentModel(m.MODEL_SPECS["herbert_twitter"], fake)
    assert [m.top_label(s) for s in model.predict(["dobre", "złe"])] == ["positive", "negative"]


def test_sentiment_model_sorts_by_length_but_keeps_input_order():
    # sortowanie po dlugosci zmniejsza dopelnianie (padding) w paczkach na GPU
    seen = []

    def fake(texts):
        seen.extend(texts)
        return [[{"label": "positive", "score": len(t) / 100}, {"label": "neutral", "score": 0.0},
                 {"label": "negative", "score": 1 - len(t) / 100}] for t in texts]
    texts = ["ccc", "a", "bbbbbb", "dd"]
    out = m.SentimentModel(m.MODEL_SPECS["herbert_twitter"], fake).predict(texts)
    assert seen == sorted(texts, key=len)
    assert [round(s["positive"] * 100) for s in out] == [3, 1, 6, 2]


def test_pick_device_fallback(monkeypatch):
    monkeypatch.setattr(m.torch.cuda, "is_available", lambda: False)
    assert m.pick_device("auto") == "cpu"
    assert m.pick_device("cuda") == "cpu"
    assert m.pick_device("cpu") == "cpu"


@pytest.mark.slow
def test_real_models_obvious_examples():
    models = m.load_models()
    for key, model in models.items():
        for variant in m.VARIANTS:
            texts = [m.variant_text("Brawo, świetna robota, dziękuję!", variant),
                     m.variant_text("To skandal i hańba, okropne kłamstwa.", variant)]
            pos, neg = model.predict(texts)
            assert m.binary_label(pos) == "positive", (key, variant)
            assert m.binary_label(neg) == "negative", (key, variant)


@pytest.mark.slow
def test_real_models_long_text():
    models = m.load_models()
    for model in models.values():
        (scores,) = model.predict(["Bardzo dobry materiał. " * 400])
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-4)
