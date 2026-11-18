"""Wspolna nakladka na pretrenowane modele sentymentu z HuggingFace.

Kazdy model zwraca inne etykiety (3 albo 5 klas, rozna wielkosc liter).
Tu sprowadzamy je do jednego formatu: {"negative": p, "neutral": p, "positive": p}.

Dwa warianty wejscia (VARIANTS):
    raw   - oryginalny tekst po lekkim czyszczeniu, wielkosc liter zachowana
    clean - text_clean z preprocessingu (preprocessor/preprocessing.py)
"""
from __future__ import annotations

import html
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

import torch

from preprocessor.preprocessing import CommentsPreprocessor

LABELS: tuple[str, ...] = ("negative", "neutral", "positive")
VARIANTS: tuple[str, ...] = ("raw", "clean")
MAX_LENGTH = 256

_PP = CommentsPreprocessor  # wspolne regexy z preprocessingu


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    display_name: str
    label_map: dict[str, str] = field(hash=False)  # surowa etykieta (male litery) -> LABELS


MODEL_SPECS: dict[str, ModelSpec] = {
    "herbert_twitter": ModelSpec(
        "herbert_twitter", "bardsai/twitter-sentiment-pl-base", "HerBERT (tweety PL)",
        {"negative": "negative", "neutral": "neutral", "positive": "positive"},
    ),
    "polbert_politics": ModelSpec(
        "polbert_politics", "eevvgg/bert-polish-sentiment-politics", "PolBERT (polityka PL)",
        {"negative": "negative", "neutral": "neutral", "positive": "positive"},
    ),
    "distilbert_multi": ModelSpec(
        "distilbert_multi", "tabularisai/multilingual-sentiment-analysis", "DistilBERT (wielojęzyczny)",
        {"very negative": "negative", "negative": "negative", "neutral": "neutral",
         "positive": "positive", "very positive": "positive"},
    ),
}


def prepare_text(text: str | None) -> str:
    """Lekkie czyszczenie dla modeli 'cased': bez zmiany wielkosci liter."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFC", text)
    text = _PP.INVISIBLE_REGEX.sub("", text)
    text = _PP.URL_REGEX.sub("http", text)
    text = _PP.USER_REGEX.sub("@user", text)
    return _PP.SPACES_REGEX.sub(" ", text).strip()


def variant_text(text: str | None, variant: str) -> str:
    if variant == "raw":
        return prepare_text(text)
    if variant == "clean":
        return CommentsPreprocessor.clean_comment(text)
    raise ValueError(f"Nieznany wariant tekstu: {variant!r} (dozwolone: {', '.join(VARIANTS)})")


def normalize_scores(raw: list[dict], label_map: dict[str, str]) -> dict[str, float]:
    scores = dict.fromkeys(LABELS, 0.0)
    for item in raw:
        label = str(item["label"]).lower()
        if label not in label_map:
            raise ValueError(f"Nieznana etykieta modelu: {item['label']!r}")
        scores[label_map[label]] += float(item["score"])
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    return scores


def top_label(scores: dict[str, float]) -> str:
    return max(LABELS, key=lambda lab: scores[lab])


def binary_label(scores: dict[str, float]) -> str:
    return "positive" if scores["positive"] >= scores["negative"] else "negative"


def pick_device(preference: str = "auto") -> str:
    if preference in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    if preference == "cuda":
        print("Uwaga: brak GPU (CUDA) - uzywam CPU.", file=sys.stderr)
    return "cpu"


class SentimentModel:
    def __init__(self, spec: ModelSpec, classifier: Callable[[list[str]], list[list[dict]]]):
        self.spec = spec
        self.classifier = classifier

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        if not texts:
            return []
        texts = list(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        raw = self.classifier([texts[i] for i in order])
        out: list[dict[str, float]] = [{}] * len(texts)
        for pos, i in enumerate(order):
            out[i] = normalize_scores(raw[pos], self.spec.label_map)
        return out


def load_models(keys: list[str] | None = None, device: str = "auto", batch_size: int = 32) -> dict[str, SentimentModel]:
    from transformers import pipeline

    dev = pick_device(device)
    models = {}
    for key in keys or list(MODEL_SPECS):
        spec = MODEL_SPECS[key]
        clf = pipeline(
            "text-classification", model=spec.hf_id, top_k=None, truncation=True,
            max_length=MAX_LENGTH, device=0 if dev == "cuda" else -1, batch_size=batch_size,
        )
        models[key] = SentimentModel(spec, clf)
    return models
