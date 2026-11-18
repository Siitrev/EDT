"""Testy predykcji wsadowej na podstawionych modelach (bez sieci i GPU)."""
import csv

import pandas as pd

from sentiment import models as m
from sentiment import predict as p

HEADER = ["comment_id", "channel_handle", "video_id", "title", "is_reply", "parent_id",
          "author_hash", "like_count", "reply_count", "published_at", "text"]


def write_comments(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for cid, ch, text in rows:
            w.writerow([cid, ch, "v1", "Tytuł", 0, "", "h", 3, 0, "2026-01-01T00:00:00Z", text])


def write_clean(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["comment_id", "text_clean"])
        w.writerows(rows)


class CountingFake:
    def __init__(self):
        self.seen = []

    def __call__(self, texts):
        self.seen.extend(texts)
        return [[{"label": "positive", "score": 0.8}, {"label": "neutral", "score": 0.1},
                 {"label": "negative", "score": 0.1}] for _ in texts]


def fake_models():
    return {k: m.SentimentModel(m.MODEL_SPECS["herbert_twitter"], CountingFake()) for k in ("a", "b")}


def test_load_comments_keeps_na_like_text(tmp_path):
    src = tmp_path / "c.csv"
    write_comments(src, [("-80mbx", "@A", "NA"), ("x2", "@A", "null")])
    df = p.load_comments(src)
    assert list(df["comment_id"]) == ["-80mbx", "x2"]
    assert list(df["text"]) == ["NA", "null"]


def test_load_comments_merges_clean_and_fills_missing(tmp_path):
    src, clean = tmp_path / "c.csv", tmp_path / "clean.csv"
    write_comments(src, [("c1", "@A", "Super FILM"), ("c2", "@A", "Nowy KOMENTARZ")])
    write_clean(clean, [("c1", "super film z pliku")])
    df = p.load_comments(src, clean)
    assert dict(zip(df["comment_id"], df["text_clean"])) == {"c1": "super film z pliku", "c2": "nowy komentarz"}


def test_load_comments_without_clean_file(tmp_path):
    src = tmp_path / "c.csv"
    write_comments(src, [("c1", "@A", "Super FILM")])
    df = p.load_comments(src, tmp_path / "brak.csv")
    assert list(df["text_clean"]) == ["super film"]


def test_select_comments_per_channel_and_limit():
    df = pd.DataFrame({"comment_id": [str(i) for i in range(10)],
                       "channel_handle": ["@A"] * 7 + ["@B"] * 3, "text": ["t"] * 10})
    assert p.select_comments(df, per_channel=5, limit=None)["channel_handle"].value_counts().to_dict() == {"@A": 5, "@B": 3}
    assert len(p.select_comments(df, per_channel=None, limit=4)) == 4
    assert len(p.select_comments(df, None, None)) == 10


def test_run_skips_empty_and_writes_long_format(tmp_path):
    src, out = tmp_path / "c.csv", tmp_path / "pred.csv"
    write_comments(src, [("c1", "@A", "Super!"), ("c2", "@A", "   "), ("c3", "@B", "Słabo")])
    counts = p.run(p.load_comments(src), fake_models(), out)
    assert counts == {"a/raw": 2, "a/clean": 2, "b/raw": 2, "b/clean": 2}
    pred = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert list(pred.columns) == p.PRED_COLUMNS
    assert set(pred["comment_id"]) == {"c1", "c3"}
    assert set(pred["variant"]) == {"raw", "clean"}


def test_run_resumes_without_duplicates(tmp_path):
    src, out = tmp_path / "c.csv", tmp_path / "pred.csv"
    write_comments(src, [(f"c{i}", "@A", f"tekst {i}") for i in range(5)])
    comments = p.load_comments(src)
    p.run(comments.head(2), fake_models(), out)
    models = fake_models()
    counts = p.run(comments, models, out, chunk_size=2)
    assert counts == {"a/raw": 3, "a/clean": 3, "b/raw": 3, "b/clean": 3}
    assert len(models["a"].classifier.seen) == 6
    pred = pd.read_csv(out, dtype=str)
    assert not pred.duplicated(["comment_id", "model", "variant"]).any()
    assert p.done_ids(out, "a", "clean") == {f"c{i}" for i in range(5)}


def test_run_repairs_torn_last_line(tmp_path):
    # przerwanie (Ctrl+C) w trakcie zapisu zostawia urwana ostatnia linie bez '\n'
    src, out = tmp_path / "c.csv", tmp_path / "pred.csv"
    write_comments(src, [("c1", "@A", "Super"), ("c2", "@A", "Dobre")])
    out.write_text(",".join(p.PRED_COLUMNS) + "\nc1,a,raw,positive,0.1,0.1,0.8\nc2,a,raw,posi", encoding="utf-8")
    p.run(p.load_comments(src), {"a": fake_models()["a"]}, out, variants=("raw",))
    pred = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert set(pred["label"]) == {"positive"}
    assert sorted(pred["comment_id"]) == ["c1", "c2"]


def test_done_ids_missing_file(tmp_path):
    assert p.done_ids(tmp_path / "nope.csv", "a", "raw") == set()
