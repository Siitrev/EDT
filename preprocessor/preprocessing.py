"""Czyszczenie i lematyzacja komentarzy z YouTube.

Uruchomienie (oba etapy):
    python preprocessor/preprocessing.py
    python preprocessor/preprocessing.py --input data/comments.csv

Mozna uruchomic tylko wybrany etap:
    python preprocessor/preprocessing.py --stage clean
    python preprocessor/preprocessing.py --stage lemma

Wejscie:  data/comments.csv       (oryginal pozostaje bez zmian)
Wyjscie:  data/comments_clean.csv (comment_id, text_clean)
          data/comments_lemma.csv (comment_id, text_lemma)

Pliki wynikowe trafiaja do tego samego katalogu co plik wejsciowy.

Instalacja:
    python -m pip install pandas spacy
    python -m spacy download pl_core_news_sm
"""

import argparse
import html
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd


class CommentsPreprocessor:
    """Przygotowuje dwa warianty tekstu do analizy sentymentu."""

    URL_REGEX = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
    USER_REGEX = re.compile(r"(?<!\w)@[\w]+(?:[.-][\w]+)*")
    PUNCT_REGEX = re.compile(r"([!?])\1{3,}")
    SPACES_REGEX = re.compile(r"\s+")
    INVISIBLE_REGEX = re.compile(r"[\u200b\ufeff]")

    def __init__(self, input_file: Path, model_name: str = "pl_core_news_sm"):
        self.input_file = Path(input_file)
        self.clean_file = self.input_file.with_name("comments_clean.csv")
        self.lemma_file = self.input_file.with_name("comments_lemma.csv")
        self.model_name = model_name

    @classmethod
    def clean_comment(cls, text: str) -> str:
        """Lekkie czyszczenie bez usuwania stopwords, emoji i interpunkcji."""
        if pd.isna(text):
            return ""

        text = html.unescape(str(text))
        text = unicodedata.normalize("NFC", text)
        text = cls.INVISIBLE_REGEX.sub("", text)
        text = cls.URL_REGEX.sub(" URL ", text)
        text = cls.USER_REGEX.sub(" USER ", text)
        text = cls.PUNCT_REGEX.sub(r"\1\1\1", text)
        text = text.lower()
        return cls.SPACES_REGEX.sub(" ", text).strip()

    @staticmethod
    def lemmatize_document(doc) -> str:
        """Zamienia slowa na lematy, zachowujac interpunkcje i emoji."""
        return "".join(
            (token.lemma_.lower() if token.is_alpha and token.lemma_ else token.text)
            + token.whitespace_
            for token in doc
        ).strip()

    def clean_csv(self) -> None:
        """Czyta oryginalny CSV i zapisuje comment_id oraz text_clean."""
        if not self.input_file.is_file():
            raise FileNotFoundError(f"Brak pliku wejściowego: {self.input_file}")

        df = pd.read_csv(
            self.input_file,
            usecols=["comment_id", "video_id", "text"],
            dtype={"comment_id": "string", "video_id": "string", "text": "string"},
            encoding="utf-8-sig",
            keep_default_na=False,
        )
        print(f"Wczytano komentarzy: {len(df)}")

        df["text_clean"] = df["text"].map(self.clean_comment)
        empty_texts = int((df["text_clean"] == "").sum())
        empty_ids = int((df["comment_id"].str.strip() == "").sum())

        df = df.loc[
            (df["text_clean"] != "") & (df["comment_id"].str.strip() != "")
        ].copy()

        duplicate_ids = int(df["comment_id"].duplicated().sum())
        if duplicate_ids:
            raise ValueError(
                f"Wykryto {duplicate_ids} powtórzonych comment_id. "
                "Identyfikatory muszą być unikalne."
            )

        # Powtarzajace sie tresci zostawiamy: moga byc wazne w monitoringu opinii.
        duplicate_texts = int(df.duplicated(["video_id", "text_clean"]).sum())

        df[["comment_id", "text_clean"]].to_csv(
            self.clean_file, index=False, encoding="utf-8-sig"
        )

        word_counts = df["text_clean"].str.split().str.len()
        char_counts = df["text_clean"].str.len()
        print(f"Puste teksty (odrzucone): {empty_texts}")
        print(f"Puste comment_id (odrzucone): {empty_ids}")
        print(f"Powtarzające się teksty w ramach filmu (pozostawione): {duplicate_texts}")
        print(f"Komentarzy po czyszczeniu: {len(df)}")
        if len(df):
            print(f"Średnia liczba słów: {word_counts.mean():.2f}")
            print(f"Średnia liczba znaków: {char_counts.mean():.2f}")

        print("\nPrzykłady czyszczenia:")
        for row in df.head(5).itertuples():
            print(f"  PRZED: {row.text}")
            print(f"  PO:    {row.text_clean}\n")
        print(f"Zapisano: {self.clean_file}")

    def lemmatize_csv(self) -> None:
        """Czyta comments_clean.csv i zapisuje comment_id oraz text_lemma."""
        if not self.clean_file.is_file():
            raise FileNotFoundError(
                f"Brak pliku {self.clean_file}. Najpierw wykonaj etap clean."
            )

        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError("Zainstaluj spaCy: python -m pip install spacy") from exc

        try:
            # Potrzebujemy morfologii do lematyzacji, dlatego nie wylaczamy jej.
            nlp = spacy.load(self.model_name, disable=["parser", "ner"])
        except OSError as exc:
            raise RuntimeError(
                f"Brak modelu {self.model_name}. Uruchom: "
                f"python -m spacy download {self.model_name}"
            ) from exc

        df = pd.read_csv(
            self.clean_file,
            usecols=["comment_id", "text_clean"],
            dtype={"comment_id": "string", "text_clean": "string"},
            encoding="utf-8-sig",
            keep_default_na=False,
        )

        print(f"\nLematyzacja {len(df)} komentarzy (model: {self.model_name})...")
        with self.lemma_file.open("w", newline="", encoding="utf-8-sig") as file:
            import csv

            writer = csv.writer(file)
            writer.writerow(["comment_id", "text_lemma"])

            docs = nlp.pipe(df["text_clean"], batch_size=64)
            for index, (comment_id, doc) in enumerate(
                zip(df["comment_id"], docs), start=1
            ):
                lemma = self.lemmatize_document(doc)
                writer.writerow([comment_id, lemma])

                if index <= 5:
                    print(f"  PRZED: {doc.text}")
                    print(f"  PO:    {lemma}\n")
                if index % 5000 == 0:
                    print(f"Zlematyzowano: {index}/{len(df)}", flush=True)

        print(f"Zapisano: {self.lemma_file} ({len(df)} komentarzy)")

    def run(self, stage: str = "all") -> None:
        """Uruchamia czyszczenie, lematyzacje lub oba etapy."""
        if stage in ("all", "clean"):
            self.clean_csv()
        if stage in ("all", "lemma"):
            self.lemmatize_csv()
        print("\nOryginalny comments.csv pozostał bez zmian.")


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    # Obsluga uruchomienia zarowno z katalogu z CSV, jak i z katalogu projektu.
    default_input = Path(__file__).resolve().parent.parent / "data" / "comments.csv"
    parser = argparse.ArgumentParser(description="Preprocessing komentarzy YouTube")
    parser.add_argument("--input", type=Path, default=default_input, help="ścieżka do comments.csv")
    parser.add_argument(
        "--stage", choices=("all", "clean", "lemma"), default="all",
        help="all: oba etapy, clean: tylko czyszczenie, lemma: tylko lematyzacja",
    )
    args = parser.parse_args(argv)

    processor = CommentsPreprocessor(args.input)
    processor.run(args.stage)


if __name__ == "__main__":
    main()
