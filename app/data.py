from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentiment.models import MODEL_SPECS, VARIANTS, LABELS, SentimentModel, binary_label, top_label, variant_text
from sentiment.predict import PRED_COLUMNS

PROB_COLUMNS = ["p_negative", "p_neutral", "p_positive"]
OUT_COLUMNS = ["model", "display_name", "variant", "label", "binary", *PROB_COLUMNS, "source"]
EMPTY_MESSAGE = "Pusty komentarz – brak predykcji"


def load_predictions(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame({c: pd.Series(dtype="float64" if c in PROB_COLUMNS else "object") for c in PRED_COLUMNS})
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", usecols=PRED_COLUMNS)
    df = df[df["label"].isin(LABELS) & df["variant"].isin(VARIANTS) & df["model"].isin(list(MODEL_SPECS))]
    for c in PROB_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ("model", "variant", "label"):
        df[c] = df[c].astype("category")
    return df[PRED_COLUMNS].reset_index(drop=True)


def filter_comments(df: pd.DataFrame, channel: str | None = None, query: str = "", only_predicted: bool = False,
                    predicted_ids: set[str] | None = None, max_rows: int = 200) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if channel:
        mask &= df["channel_handle"] == channel
    if query:
        mask &= df["text"].str.contains(query, case=False, regex=False, na=False)
    if only_predicted:
        mask &= df["comment_id"].isin(predicted_ids or set())
    return df[mask].nlargest(max_rows, "like_count", keep="first")


def _row(key: str, variant: str, scores: dict[str, float], source: str) -> dict:
    return {
        "model": key, "display_name": MODEL_SPECS[key].display_name, "variant": variant,
        "label": top_label(scores), "binary": binary_label(scores),
        "p_negative": scores["negative"], "p_neutral": scores["neutral"], "p_positive": scores["positive"],
        "source": source,
    }


def predictions_for(comment_id: str | None, text: str, models: dict[str, SentimentModel] | None,
                    cached: pd.DataFrame, variants: tuple[str, ...] = VARIANTS) -> pd.DataFrame:
    """Predykcje wszystkich modeli dla jednego tekstu: najpierw z pliku, brakujace na zywo."""
    if not variant_text(text, "raw"):
        return pd.DataFrame(columns=OUT_COLUMNS)
    cache: dict[tuple[str, str], dict[str, float]] = {}
    if comment_id is not None and not cached.empty:
        for r in cached[cached["comment_id"] == comment_id].itertuples(index=False):
            cache[(r.model, r.variant)] = {"negative": float(r.p_negative), "neutral": float(r.p_neutral),
                                           "positive": float(r.p_positive)}
    rows = []
    for key in MODEL_SPECS:
        for variant in variants:
            if (key, variant) in cache:
                rows.append(_row(key, variant, cache[(key, variant)], "plik"))
            elif models and key in models:
                t = variant_text(text, variant)
                if t:
                    rows.append(_row(key, variant, models[key].predict([t])[0], "na żywo"))
    return pd.DataFrame(rows, columns=OUT_COLUMNS)


def agreement(preds: pd.DataFrame) -> str:
    parts = []
    for variant in VARIANTS:
        values = set(preds.loc[preds["variant"] == variant, "binary"])
        if not values:
            continue
        parts.append(f"{variant}: zgodne – {values.pop()}" if len(values) == 1 else f"{variant}: niezgodne")
    return " | ".join(parts)


def channel_summary(preds: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    cols = ["channel_handle", "model", "variant", "n", *LABELS]
    if preds.empty:
        return pd.DataFrame(columns=cols)
    merged = preds.merge(comments[["comment_id", "channel_handle"]], on="comment_id", how="inner")
    keys = ["channel_handle", "model", "variant"]
    shares = pd.crosstab([merged[k] for k in keys], merged["label"], normalize="index")
    shares = shares.reindex(columns=list(LABELS), fill_value=0.0)
    counts = merged.groupby(keys).size().rename("n")
    out = shares.join(counts).reset_index()
    out.columns.name = None
    return out[cols]
