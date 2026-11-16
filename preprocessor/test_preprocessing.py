"""Testy preprocessingu komentarzy YouTube (pytest).

Testy pracuja na plikach tymczasowych, bez dostepu do prawdziwego data/.
Lematyzator jest zastepowany sztucznym modelem, wiec testy nie pobieraja spaCy.

Uruchomienie z katalogu glownego repo:
    python -m pytest preprocessor/test_preprocessing.py -v
"""

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).resolve().with_name("preprocessing.py")
spec = importlib.util.spec_from_file_location("preprocessing", SCRIPT_PATH)
preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocessing)
CommentsPreprocessor = preprocessing.CommentsPreprocessor


def save_comments(path, rows):
    """Zapisuje niewielki, sztuczny plik o strukturze komentarzy z YouTube."""
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["comment_id", "video_id", "text"])
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def test_clean_comment_normalization():
    text = "  ZaŻÓŁĆ\u200b &amp;  SPRAWDŹ\n TO!!!!!!  "
    assert CommentsPreprocessor.clean_comment(text) == "zażółć & sprawdź to!!!"


def test_clean_comment_urls_users_and_emoji():
    text = "@JanKowalski NIE zgadzam się 😂 https://example.com !!!!!!"
    assert CommentsPreprocessor.clean_comment(text) == "user nie zgadzam się 😂 url !!!"


def test_clean_comment_keeps_negations_and_punctuation():
    assert CommentsPreprocessor.clean_comment("NIE! Nigdy bez tego ❤️") == "nie! nigdy bez tego ❤️"


@pytest.mark.parametrize("text", [None, "", "  \t\n  ", "\u200b\ufeff"])
def test_clean_comment_empty(text):
    assert CommentsPreprocessor.clean_comment(text) == ""


def test_clean_csv_outputs_only_two_columns_without_changing_original(tmp_path):
    source = tmp_path / "comments.csv"
    save_comments(source, [
        ("c1", "video1", "  SUPER film!!!!! 😂  "),
        ("c2", "video1", "SUPER film!!!!! 😂"),  # powtorzona tresc ma pozostac
        ("c3", "video2", "  "),              # pusty komentarz
        ("", "video2", "dobry film"),          # brak comment_id
        ("c4", "video2", "Nie zgadzam się!"),
    ])
    original_bytes = source.read_bytes()

    processor = CommentsPreprocessor(source)
    processor.clean_csv()
    result = read_csv(processor.clean_file)

    assert source.read_bytes() == original_bytes
    assert list(result[0]) == ["comment_id", "text_clean"]
    assert [r["comment_id"] for r in result] == ["c1", "c2", "c4"]
    assert [r["text_clean"] for r in result] == [
        "super film!!! 😂", "super film!!! 😂", "nie zgadzam się!"
    ]
    assert not processor.lemma_file.exists()  # etap clean nie uruchamia lematyzacji


def test_clean_csv_rejects_duplicate_comment_ids(tmp_path):
    source = tmp_path / "comments.csv"
    save_comments(source, [
        ("same_id", "video1", "pierwszy komentarz"),
        ("same_id", "video2", "drugi komentarz"),
    ])
    processor = CommentsPreprocessor(source)

    with pytest.raises(ValueError, match="comment_id"):
        processor.clean_csv()
    assert not processor.clean_file.exists()


def test_clean_csv_missing_source(tmp_path):
    processor = CommentsPreprocessor(tmp_path / "comments.csv")
    with pytest.raises(FileNotFoundError):
        processor.clean_csv()


def test_lemmatize_document_preserves_punctuation_and_emoji():
    # Udajemy tokeny zwracane przez spaCy.
    tokens = [
        SimpleNamespace(text="koty", lemma_="kot", is_alpha=True, whitespace_=" "),
        SimpleNamespace(text="biegały", lemma_="biegać", is_alpha=True, whitespace_=""),
        SimpleNamespace(text="!", lemma_="!", is_alpha=False, whitespace_=" "),
        SimpleNamespace(text="😂", lemma_="", is_alpha=False, whitespace_=""),
    ]
    assert CommentsPreprocessor.lemmatize_document(tokens) == "kot biegać! 😂"


def test_lemmatize_csv_from_clean_file_without_real_spacy_model(tmp_path, monkeypatch):
    source = tmp_path / "comments.csv"
    save_comments(source, [
        ("c1", "video1", "Koty biegały! 😂"),
        ("c2", "video2", "Nie lubię!"),
    ])
    processor = CommentsPreprocessor(source)
    processor.clean_csv()
    original_bytes = source.read_bytes()
    clean_bytes = processor.clean_file.read_bytes()

    class FakeDocument:
        def __init__(self, text, tokens):
            self.text = text
            self.tokens = tokens

        def __iter__(self):
            return iter(self.tokens)

    def token(text, lemma, space=""):
        return SimpleNamespace(
            text=text, lemma_=lemma, is_alpha=text.isalpha(), whitespace_=space
        )

    class FakeNLP:
        def pipe(self, texts, batch_size=64):
            for text in texts:
                if text == "koty biegały! 😂":
                    yield FakeDocument(text, [
                        token("koty", "kot", " "),
                        token("biegały", "biegać"),
                        token("!", "!", " "),
                        token("😂", ""),
                    ])
                elif text == "nie lubię!":
                    yield FakeDocument(text, [
                        token("nie", "nie", " "),
                        token("lubię", "lubić"),
                        token("!", "!"),
                    ])
                else:
                    raise AssertionError(f"Nieoczekiwany tekst: {text}")

    import spacy
    monkeypatch.setattr(spacy, "load", lambda *args, **kwargs: FakeNLP())

    processor.lemmatize_csv()
    rows = read_csv(processor.lemma_file)

    assert list(rows[0]) == ["comment_id", "text_lemma"]
    assert rows == [
        {"comment_id": "c1", "text_lemma": "kot biegać! 😂"},
        {"comment_id": "c2", "text_lemma": "nie lubić!"},
    ]
    assert source.read_bytes() == original_bytes
    assert processor.clean_file.read_bytes() == clean_bytes


def test_lemmatize_csv_requires_clean_file(tmp_path):
    processor = CommentsPreprocessor(tmp_path / "comments.csv")
    with pytest.raises(FileNotFoundError, match="clean"):
        processor.lemmatize_csv()
