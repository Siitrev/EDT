import pandas as pd
import pytest

from app import data as d
from sentiment import models as m
from sentiment.predict import PRED_COLUMNS


def comments_df():
    return pd.DataFrame({
        "comment_id": ["c1", "c2", "c3"], "channel_handle": ["@A", "@A", "@B"],
        "video_id": ["v", "v", "w"], "title": ["T", "T", "U"], "like_count": [1, 50, 7],
        "published_at": ["2026"] * 3, "text": ["Świetny film", "Kłamstwa i HAŃBA", "nic"],
        "text_clean": ["świetny film", "kłamstwa i hańba", "nic"]})


def preds_df():
    rows = [("c1", "herbert_twitter", "raw", "positive", 0.1, 0.1, 0.8),
            ("c1", "herbert_twitter", "clean", "positive", 0.2, 0.1, 0.7),
            ("c1", "polbert_politics", "raw", "positive", 0.2, 0.1, 0.7),
            ("c3", "herbert_twitter", "raw", "neutral", 0.2, 0.6, 0.2)]
    return pd.DataFrame(rows, columns=PRED_COLUMNS)


def fake_models():
    def fake(texts):
        return [[{"label": "negative", "score": 1.0}, {"label": "neutral", "score": 0.0},
                 {"label": "positive", "score": 0.0}] for _ in texts]
    return {k: m.SentimentModel(m.MODEL_SPECS["herbert_twitter"], fake) for k in m.MODEL_SPECS}


def test_load_predictions_missing_file(tmp_path):
    out = d.load_predictions(tmp_path / "brak.csv")
    assert out.empty and list(out.columns) == PRED_COLUMNS


def test_load_predictions_drops_corrupted_rows(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text(",".join(PRED_COLUMNS) + "\nc1,herbert_twitter,raw,positive,0.1,0.1,0.8\n"
                    "c2,herbert_twitter,raw,positivc3,herbert_twitter,raw,positive,0.1,0.1,0.8\n", encoding="utf-8")
    out = d.load_predictions(path)
    assert list(out["comment_id"]) == ["c1"]


def test_filter_comments_channel_query_and_order():
    df = comments_df()
    assert list(d.filter_comments(df)["comment_id"]) == ["c2", "c3", "c1"]
    assert list(d.filter_comments(df, channel="@A")["comment_id"]) == ["c2", "c1"]
    assert list(d.filter_comments(df, query="hańba")["comment_id"]) == ["c2"]
    assert list(d.filter_comments(df, query="(")["comment_id"]) == []
    assert list(d.filter_comments(df, only_predicted=True, predicted_ids={"c1"})["comment_id"]) == ["c1"]
    assert len(d.filter_comments(df, max_rows=1)) == 1


def test_predictions_for_uses_cache_then_live():
    out = d.predictions_for("c1", "Świetny film", fake_models(), preds_df())
    keys = list(zip(out["model"], out["variant"]))
    assert keys == [(k, v) for k in m.MODEL_SPECS for v in m.VARIANTS]
    assert list(out["source"]) == ["plik", "plik", "plik", "na żywo", "na żywo", "na żywo"]
    assert list(out["binary"])[-1] == "negative"


def test_predictions_for_without_models_returns_only_cached():
    out = d.predictions_for("c1", "Świetny film", None, preds_df())
    assert list(zip(out["model"], out["variant"])) == [
        ("herbert_twitter", "raw"), ("herbert_twitter", "clean"), ("polbert_politics", "raw")]


def test_predictions_for_empty_text():
    assert d.predictions_for(None, "  ​ ", fake_models(), preds_df()).empty


def test_agreement_per_variant():
    cached = d.predictions_for("c1", "x", None, preds_df())
    assert d.agreement(cached) == "raw: zgodne – positive | clean: zgodne – positive"
    mixed = d.predictions_for("c1", "x", fake_models(), preds_df())
    assert d.agreement(mixed) == "raw: niezgodne | clean: niezgodne"


def test_channel_summary():
    s = d.channel_summary(preds_df(), comments_df())
    row = s[(s.channel_handle == "@A") & (s.model == "herbert_twitter") & (s.variant == "raw")].iloc[0]
    assert row.n == 1 and row.positive == pytest.approx(1.0) and row.negative == 0
