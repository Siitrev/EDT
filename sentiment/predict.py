"""Wsadowa predykcja sentymentu dla zebranych komentarzy YouTube.

Uzycie (z katalogu projektu):
    python -m sentiment.predict                      # wszystkie komentarze, 3 modele x 2 warianty
    python -m sentiment.predict --per-channel 200    # probka na kanal (szybki test)
    python -m sentiment.predict --models herbert_twitter --variants raw

Wynik: data/predictions.csv w formacie dlugim
    comment_id, model, variant, label, p_negative, p_neutral, p_positive
Skrypt mozna przerwac i uruchomic ponownie - ocenione pary (komentarz, model, wariant)
sa pomijane.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from sentiment.models import MODEL_SPECS, VARIANTS, SentimentModel, load_models, prepare_text, top_label, variant_text

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PRED_COLUMNS = ["comment_id", "model", "variant", "label", "p_negative", "p_neutral", "p_positive"]


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", **kwargs)


def load_comments(path: Path, clean_path: Path | None = None) -> pd.DataFrame:
    """Wczytuje komentarze i dokleja text_clean (z pliku albo liczony na biezaco)."""
    df = _read_csv(Path(path))
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)
    df["text_clean"] = pd.NA
    if clean_path is not None and Path(clean_path).is_file():
        clean = _read_csv(Path(clean_path), usecols=["comment_id", "text_clean"]).drop_duplicates("comment_id")
        df["text_clean"] = df["comment_id"].map(clean.set_index("comment_id")["text_clean"])
    missing = df["text_clean"].isna()
    if missing.any():
        df.loc[missing, "text_clean"] = df.loc[missing, "text"].map(lambda t: variant_text(t, "clean"))
        print(f"text_clean uzupelniony na biezaco dla {int(missing.sum())} komentarzy", flush=True)
    df["text_clean"] = df["text_clean"].astype(str)
    return df


def select_comments(df: pd.DataFrame, per_channel: int | None, limit: int | None, seed: int = 42) -> pd.DataFrame:
    if per_channel is not None:
        picked = [
            g.sample(n=min(per_channel, len(g)), random_state=seed).index
            for _, g in df.groupby("channel_handle", sort=False)
        ]
        df = df.loc[[i for idx in picked for i in idx]]
    if limit is not None:
        df = df.head(limit)
    return df


def _done_map(pred_path: Path) -> dict[tuple[str, str], set[str]]:
    if not Path(pred_path).is_file():
        return {}
    pred = _read_csv(Path(pred_path), usecols=["comment_id", "model", "variant"])
    return {key: set(g["comment_id"]) for key, g in pred.groupby(["model", "variant"])}


def done_ids(pred_path: Path, model_key: str, variant: str) -> set[str]:
    return _done_map(pred_path).get((model_key, variant), set())


def _variant_series(comments: pd.DataFrame, variant: str) -> pd.Series:
    if variant == "raw":
        return comments["text"].map(prepare_text)
    if variant == "clean":
        return comments["text_clean"].fillna("").astype(str).str.strip()
    raise ValueError(f"Nieznany wariant tekstu: {variant!r}")


def repair_torn_tail(path: Path) -> None:
    """Obcina urwana ostatnia linie (np. po Ctrl+C w trakcie zapisu), zeby dopisywanie jej nie sklejalo."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open("rb+") as f:
        f.seek(-1, 2)
        if f.read(1) == b"\n":
            return
        pos = f.seek(0, 2)
        while pos > 0:
            start = max(0, pos - 65536)
            f.seek(start)
            cut = f.read(pos - start).rfind(b"\n")
            if cut != -1:
                f.truncate(start + cut + 1)
                print(f"Usunieto urwana ostatnia linie w {path.name}", flush=True)
                return
            pos = start
        f.truncate(0)


def run(comments: pd.DataFrame, models: dict[str, SentimentModel], pred_path: Path,
        variants: tuple[str, ...] = VARIANTS, chunk_size: int = 2048) -> dict[str, int]:
    pred_path = Path(pred_path)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    repair_torn_tail(pred_path)
    done = _done_map(pred_path)
    texts = {v: _variant_series(comments, v) for v in variants}
    counts: dict[str, int] = {}
    for key, model in models.items():
        for variant in variants:
            series = texts[variant]
            seen = done.get((key, variant), set())
            mask = (series != "") & ~comments["comment_id"].isin(seen)
            ids, todo = comments.loc[mask, "comment_id"].tolist(), series[mask].tolist()
            tag = f"{key}/{variant}"
            counts[tag] = 0
            start = time.time()
            for i in range(0, len(todo), chunk_size):
                scores = model.predict(todo[i:i + chunk_size])
                rows = [
                    (cid, key, variant, top_label(s), round(s["negative"], 4), round(s["neutral"], 4), round(s["positive"], 4))
                    for cid, s in zip(ids[i:i + chunk_size], scores)
                ]
                pd.DataFrame(rows, columns=PRED_COLUMNS).to_csv(
                    pred_path, mode="a", header=not pred_path.is_file() or pred_path.stat().st_size == 0,
                    index=False, encoding="utf-8",
                )
                counts[tag] += len(rows)
                rate = counts[tag] / max(time.time() - start, 1e-9)
                print(f"[{tag}] {counts[tag]:,}/{len(todo):,} ({rate:,.0f} kom./s)".replace(",", " "), flush=True)
            if not todo:
                print(f"[{tag}] nic do zrobienia", flush=True)
    return counts


def summary(pred_path: Path, comments: pd.DataFrame) -> pd.DataFrame:
    pred = _read_csv(Path(pred_path), usecols=["comment_id", "model", "variant", "label"])
    pred = pred.merge(comments[["comment_id", "channel_handle"]], on="comment_id", how="inner")
    table = pd.crosstab([pred["channel_handle"], pred["model"], pred["variant"]], pred["label"], normalize="index")
    return (table * 100).round(1)


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Predykcja sentymentu komentarzy (3 pretrenowane modele).")
    ap.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=list(MODEL_SPECS))
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    ap.add_argument("--per-channel", type=int, default=None, help="losowa probka N komentarzy na kanal")
    ap.add_argument("--limit", type=int, default=None, help="maksymalnie N komentarzy")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--input", type=Path, default=DATA / "comments.csv")
    ap.add_argument("--clean-input", type=Path, default=DATA / "comments_clean.csv")
    ap.add_argument("--output", type=Path, default=DATA / "predictions.csv")
    args = ap.parse_args(argv)

    if not args.clean_input.is_file():
        print(f"Brak {args.clean_input} - text_clean zostanie policzony na biezaco.", flush=True)
    comments = load_comments(args.input, args.clean_input)
    comments = select_comments(comments, args.per_channel, args.limit)
    print(f"Komentarzy do oceny: {len(comments):,}".replace(",", " "), flush=True)
    models = load_models(args.models, device=args.device, batch_size=args.batch_size)
    try:
        counts = run(comments, models, args.output, variants=tuple(args.variants))
        print("Nowe predykcje:", counts)
    except KeyboardInterrupt:
        print("\nPrzerwano. Zapisane predykcje zostaja; uruchom ponownie, aby kontynuowac.")
    if args.output.is_file():
        print("\nRozklad etykiet [%] (kanal, model, wariant):")
        print(summary(args.output, comments).to_string())


if __name__ == "__main__":
    main()
