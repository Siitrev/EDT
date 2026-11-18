"""Monitor opinii - okienkowa aplikacja tkinter porownujaca modele sentymentu.

Uruchomienie z katalogu projektu:
    python -m app.gui

Zakladki:
    Komentarze   - filtr kanalu, wyszukiwarka, lista komentarzy; po wybraniu komentarza
                   predykcje 3 modeli x 2 warianty tekstu (raw / clean) + wykres
    Wlasny tekst - dowolne zdanie oceniane na zywo przez wszystkie modele
    Podsumowanie - rozklad sentymentu per kanal, model i wariant (z data/predictions.csv)

Predykcje sa brane z data/predictions.csv (python -m sentiment.predict); brakujace
liczone na zywo. Ciezkie operacje (wczytanie ~1 mln komentarzy, ladowanie modeli,
predykcje) ida w watku roboczym, zeby okno nie zamarzalo.
"""
from __future__ import annotations

import queue
import random
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from app.data import EMPTY_MESSAGE, agreement, channel_summary, filter_comments, load_predictions, predictions_for
from sentiment.models import LABELS, MODEL_SPECS, VARIANTS, SentimentModel, load_models, variant_text
from sentiment.predict import load_comments

ROOT = Path(__file__).resolve().parent.parent
ALL_CHANNELS = "Wszystkie"
COLORS = {"negative": "#d64545", "neutral": "#9aa0a6", "positive": "#3a9d5d"}
LABELS_PL = {"negative": "negatywny", "neutral": "neutralny", "positive": "pozytywny"}
MAX_ROWS = 200


def _fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


class PredictionPanel(ttk.Frame):
    """Tabela predykcji + wykres skumulowanych prawdopodobienstw + zgodnosc modeli."""

    COLUMNS = (("display_name", "Model", 190), ("variant", "Wariant", 70), ("label", "Etykieta", 80),
               ("binary", "Binarnie", 80), ("p_negative", "P(neg)", 65), ("p_neutral", "P(neu)", 65),
               ("p_positive", "P(poz)", 65), ("source", "Źródło", 70))

    def __init__(self, master):
        super().__init__(master)
        self.token = 0
        self.tree = ttk.Treeview(self, columns=[c[0] for c in self.COLUMNS], show="headings", height=6)
        for key, title, width in self.COLUMNS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor="w" if key == "display_name" else "center", stretch=key == "display_name")
        self.tree.pack(fill="x")
        self.agreement_var = tk.StringVar()
        ttk.Label(self, textvariable=self.agreement_var, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 2))
        self.figure = Figure(figsize=(7.5, 2.4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def clear(self, message: str = "") -> None:
        self.token += 1  # wyniki zlecone przed wyczyszczeniem sa juz nieaktualne
        self.tree.delete(*self.tree.get_children())
        self.agreement_var.set(message)
        self.figure.clear()
        self.canvas.draw_idle()

    def render(self, preds: pd.DataFrame) -> None:
        if preds.empty:
            self.clear(EMPTY_MESSAGE)
            return
        self.tree.delete(*self.tree.get_children())
        for r in preds.itertuples(index=False):
            self.tree.insert("", "end", values=(
                r.display_name, r.variant, LABELS_PL[r.label], LABELS_PL[r.binary],
                f"{r.p_negative:.2f}", f"{r.p_neutral:.2f}", f"{r.p_positive:.2f}", r.source))
        self.agreement_var.set(agreement(preds))
        self._draw(preds)

    def _draw(self, preds: pd.DataFrame) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        names = [f"{r.display_name} · {r.variant}" for r in preds.itertuples(index=False)]
        left = [0.0] * len(preds)
        for label in LABELS:
            values = preds[f"p_{label}"].astype(float).tolist()
            ax.barh(names, values, left=left, color=COLORS[label], label=LABELS_PL[label])
            left = [a + b for a, b in zip(left, values)]
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.tick_params(labelsize=8)
        ax.legend(loc="lower right", fontsize=7, ncol=3)
        self.figure.tight_layout()
        self.canvas.draw_idle()


class App(tk.Tk):
    def __init__(self, comments: pd.DataFrame | None = None, predictions: pd.DataFrame | None = None,
                 models_loader: Callable[[], dict[str, SentimentModel]] = load_models,
                 comments_path: Path = ROOT / "data" / "comments.csv",
                 clean_path: Path = ROOT / "data" / "comments_clean.csv",
                 predictions_path: Path = ROOT / "data" / "predictions.csv",
                 sync: bool = False):
        super().__init__()
        self.title("Monitor opinii - YouTube")
        self.geometry("1280x900")
        self.minsize(900, 650)
        self.sync = sync
        self.models_loader = models_loader
        self.models: dict[str, SentimentModel] | None = None
        self.comments: pd.DataFrame | None = None
        self.predictions: pd.DataFrame | None = None
        self.predictions_path = Path(predictions_path)
        self._predictions_from_file = predictions is None
        self.predicted_ids: set[str] = set()
        self._pred_index: dict[str, object] = {}
        self._by_id: pd.DataFrame | None = None
        self._filtered = pd.DataFrame()
        self._current_id: str | None = None
        self._filter_token = 0
        self._models_waiting: list[tuple[Callable[[], None], Callable[[Exception], None]]] = []
        self._queue: queue.Queue = queue.Queue()
        self._after_id = None
        self._alive = True

        self.channel_var = tk.StringVar(value=ALL_CHANNELS)
        self.query_var = tk.StringVar()
        self.only_pred_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Wczytywanie komentarzy…")
        self.clean_preview_var = tk.StringVar()
        self.summary_info_var = tk.StringVar()
        self._build_ui()

        if not sync:
            self._after_id = self.after(100, self._poll)
        paths = (comments_path, clean_path, predictions_path)
        self._run(lambda: self._load_data(comments, predictions, *paths), self._on_data_loaded)

    # ------------------------------------------------------------------ budowa okna
    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        self._build_comments_tab(notebook)
        self._build_custom_tab(notebook)
        self._build_summary_tab(notebook)
        ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x", side="bottom")

    def _build_comments_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=6)
        notebook.add(tab, text="Komentarze")
        bar = ttk.Frame(tab)
        bar.pack(fill="x")
        ttk.Label(bar, text="Kanał:").pack(side="left")
        self.channel_box = ttk.Combobox(bar, textvariable=self.channel_var, values=[ALL_CHANNELS], state="readonly", width=24)
        self.channel_box.pack(side="left", padx=(4, 12))
        self.channel_box.bind("<<ComboboxSelected>>", lambda _e: self.apply_filter())
        ttk.Label(bar, text="Szukaj:").pack(side="left")
        entry = ttk.Entry(bar, textvariable=self.query_var, width=32)
        entry.pack(side="left", padx=(4, 12))
        entry.bind("<Return>", lambda _e: self.apply_filter())
        ttk.Checkbutton(bar, text="tylko z predykcjami", variable=self.only_pred_var,
                        command=self.apply_filter).pack(side="left", padx=(0, 12))
        ttk.Button(bar, text="Szukaj", command=self.apply_filter).pack(side="left")
        ttk.Button(bar, text="Losowy", command=self.random_comment).pack(side="left", padx=4)

        cols = (("channel", "Kanał", 140), ("likes", "Polubienia", 80), ("date", "Data", 90), ("text", "Treść", 800))
        frame = ttk.Frame(tab)
        frame.pack(fill="both", expand=True, pady=6)
        self.comments_tree = ttk.Treeview(frame, columns=[c[0] for c in cols], show="headings", height=9)
        for key, title, width in cols:
            self.comments_tree.heading(key, text=title)
            self.comments_tree.column(key, width=width, stretch=key == "text", anchor="w" if key == "text" else "center")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.comments_tree.yview)
        self.comments_tree.configure(yscrollcommand=scroll.set)
        self.comments_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.comments_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        self.comment_text = tk.Text(tab, height=7, wrap="word", font=("Segoe UI", 10), state="disabled")
        self.comment_text.pack(fill="x")
        self.comment_panel = PredictionPanel(tab)
        self.comment_panel.pack(fill="both", expand=True, pady=(6, 0))
        self.pred_tree = self.comment_panel.tree
        self.agreement_var = self.comment_panel.agreement_var

    def _build_custom_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=6)
        notebook.add(tab, text="Własny tekst")
        ttk.Label(tab, text="Wpisz komentarz do oceny:").pack(anchor="w")
        self.custom_text = tk.Text(tab, height=5, wrap="word", font=("Segoe UI", 10))
        self.custom_text.pack(fill="x", pady=4)
        ttk.Button(tab, text="Oceń", command=self.predict_custom).pack(anchor="w")
        ttk.Label(tab, textvariable=self.clean_preview_var, foreground="#555", wraplength=1100).pack(anchor="w", pady=4)
        self.custom_panel = PredictionPanel(tab)
        self.custom_panel.pack(fill="both", expand=True)
        self.custom_pred_tree = self.custom_panel.tree
        self.custom_agreement_var = self.custom_panel.agreement_var

    def _build_summary_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=6)
        notebook.add(tab, text="Podsumowanie")
        top = ttk.Frame(tab)
        top.pack(fill="x")
        ttk.Button(top, text="Odśwież", command=self.refresh_summary).pack(side="left")
        ttk.Label(top, textvariable=self.summary_info_var).pack(side="left", padx=8)
        cols = (("channel", "Kanał", 150), ("model", "Model", 190), ("variant", "Wariant", 70), ("n", "Liczba", 90),
                ("negative", "Negatywne", 90), ("neutral", "Neutralne", 90), ("positive", "Pozytywne", 90))
        self.summary_tree = ttk.Treeview(tab, columns=[c[0] for c in cols], show="headings", height=12)
        for key, title, width in cols:
            self.summary_tree.heading(key, text=title)
            self.summary_tree.column(key, width=width, anchor="w" if key in ("channel", "model") else "center")
        self.summary_tree.pack(fill="x", pady=6)
        self.summary_figure = Figure(figsize=(8, 3.6), dpi=100)
        self.summary_canvas = FigureCanvasTkAgg(self.summary_figure, master=tab)
        self.summary_canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------ watki
    def _run(self, work: Callable[[], object], on_done: Callable[[object], None],
             on_error: Callable[[Exception], None] | None = None) -> None:
        """Wykonuje prace w tle (albo od razu w trybie sync) i oddaje wynik do watku glownego.

        Watek roboczy nigdy nie dotyka widzetow: wynik/blad trafia do kolejki,
        ktora `_poll` oproznia w watku glownym.
        """
        if self.sync:
            try:
                result = work()
            except Exception as exc:
                if on_error is None:
                    raise
                on_error(exc)
                return
            on_done(result)
            return

        def target():
            try:
                self._queue.put((on_done, on_error, work(), None))
            except Exception as exc:
                self._queue.put((on_done, on_error, None, exc))

        threading.Thread(target=target, daemon=True).start()

    def _report_error(self, error: Exception) -> None:
        self.status_var.set(f"Błąd: {error}")
        messagebox.showerror("Błąd", str(error), parent=self)

    def _poll(self) -> None:
        try:
            while True:
                try:
                    on_done, on_error, result, error = self._queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    if error is not None:
                        (on_error or self._report_error)(error)
                    else:
                        on_done(result)
                except Exception as exc:
                    traceback.print_exc()
                    self.status_var.set(f"Błąd: {exc}")
        finally:
            if self._alive:
                self._after_id = self.after(100, self._poll)

    def destroy(self) -> None:
        self._alive = False
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
        super().destroy()

    # ------------------------------------------------------------------ dane i modele
    @staticmethod
    def _prepare_predictions(predictions: pd.DataFrame) -> dict:
        """Indeks comment_id -> pozycje wierszy (liczony raz, w watku roboczym).

        Bez niego kazde klikniecie skanowaloby caly plik predykcji (~6 mln wierszy).
        """
        index = predictions.groupby("comment_id", sort=False).indices if not predictions.empty else {}
        return {"predictions": predictions, "index": index, "ids": set(index)}

    @classmethod
    def _load_data(cls, comments, predictions, comments_path, clean_path, predictions_path) -> dict:
        if comments is None:
            comments = load_comments(comments_path, clean_path)
        if predictions is None:
            predictions = load_predictions(predictions_path)
        bundle = cls._prepare_predictions(predictions)
        bundle["comments"] = comments
        bundle["by_id"] = comments.drop_duplicates("comment_id").set_index("comment_id")
        bundle["channels"] = sorted(comments["channel_handle"].unique())
        return bundle

    def _set_predictions(self, bundle: dict) -> None:
        self.predictions = bundle["predictions"]
        self._pred_index = bundle["index"]
        self.predicted_ids = bundle["ids"]

    def _on_data_loaded(self, bundle: dict) -> None:
        self.comments = bundle["comments"]
        self._by_id = bundle["by_id"]
        self._set_predictions(bundle)
        self.channel_box.configure(values=[ALL_CHANNELS, *bundle["channels"]])
        self._update_status()
        self.apply_filter()
        self._compute_summary()

    def _cached_rows(self, comment_id: str | None) -> pd.DataFrame:
        if self.predictions is None:
            return pd.DataFrame()
        positions = self._pred_index.get(comment_id) if comment_id is not None else None
        if positions is None:
            return self.predictions.iloc[0:0]
        return self.predictions.iloc[positions]

    def _update_status(self, extra: str = "") -> None:
        if self.comments is None:
            return
        models = "załadowane" if self.models is not None else "ładowane przy pierwszej predykcji na żywo"
        text = (f"Komentarzy: {_fmt_int(len(self.comments))} | predykcji w pliku: {_fmt_int(len(self.predictions))}"
                f" | modele: {models}")
        self.status_var.set(f"{text} | {extra}" if extra else text)

    def _with_models(self, callback: Callable[[], None], on_fail: Callable[[Exception], None]) -> None:
        if self.models is not None:
            callback()
            return
        self._models_waiting.append((callback, on_fail))
        if len(self._models_waiting) > 1:
            return  # ladowanie juz trwa
        self._update_status("Ładowanie modeli…")
        self._run(self.models_loader, self._on_models_loaded, self._on_models_failed)

    def _on_models_loaded(self, models) -> None:
        self.models = models
        self._update_status()
        waiting, self._models_waiting = self._models_waiting, []
        for callback, _ in waiting:
            callback()

    def _on_models_failed(self, error: Exception) -> None:
        waiting, self._models_waiting = self._models_waiting, []
        self._update_status(f"Błąd ładowania modeli: {error}")
        for _, on_fail in waiting:
            on_fail(error)

    def _needs_models(self, comment_id: str | None) -> bool:
        have = self._cached_rows(comment_id)
        if have.empty:
            return True
        return len(have[["model", "variant"]].drop_duplicates()) < len(MODEL_SPECS) * len(VARIANTS)

    def _predict_into(self, panel: PredictionPanel, comment_id: str | None, text: str) -> None:
        if not variant_text(text, "raw"):
            panel.clear(EMPTY_MESSAGE)
            return
        panel.clear("Liczenie predykcji…")
        token = panel.token
        cached = self._cached_rows(comment_id)

        def current() -> bool:
            return panel.token == token

        def render(preds):
            if current():
                panel.render(preds)

        def fail(prefix: str):
            def handler(error: Exception):
                if current():
                    panel.clear(f"{prefix}: {error}")
            return handler

        def start():
            self._run(lambda: predictions_for(comment_id, text, self.models, cached), render, fail("Błąd predykcji"))

        if self._needs_models(comment_id):
            self._with_models(start, fail("Błąd ładowania modeli"))
        else:
            start()

    # ------------------------------------------------------------------ akcje
    def apply_filter(self) -> None:
        if self.comments is None:
            return
        self._filter_token += 1
        token = self._filter_token
        channel = self.channel_var.get()
        comments, ids = self.comments, self.predicted_ids
        params = dict(channel=None if channel == ALL_CHANNELS else channel, query=self.query_var.get().strip(),
                      only_predicted=self.only_pred_var.get(), max_rows=MAX_ROWS)
        self._run(lambda: filter_comments(comments, predicted_ids=ids, **params),
                  lambda df: self._show_filtered(df) if token == self._filter_token else None)

    def _show_filtered(self, filtered: pd.DataFrame) -> None:
        self._filtered = filtered
        tree = self.comments_tree
        tree.delete(*tree.get_children())
        for r in self._filtered.itertuples(index=False):
            short = " ".join(str(r.text).split())
            tree.insert("", "end", iid=r.comment_id, values=(
                r.channel_handle, r.like_count, str(r.published_at)[:10], short[:120] + ("…" if len(short) > 120 else "")))
        if self._filtered.empty:
            self._current_id = None
            self._set_comment_text("Brak komentarzy dla tego filtra.")
            self.comment_panel.clear()
            return
        first = self._filtered["comment_id"].iloc[0]
        self.show_comment(first)
        tree.selection_set(first)
        tree.see(first)

    def _on_tree_select(self, _event=None) -> None:
        selected = self.comments_tree.selection()
        if selected and selected[0] != self._current_id:
            self.show_comment(selected[0])

    def _set_comment_text(self, text: str) -> None:
        self.comment_text.configure(state="normal")
        self.comment_text.delete("1.0", "end")
        self.comment_text.insert("1.0", text)
        self.comment_text.configure(state="disabled")

    def show_comment(self, comment_id: str) -> None:
        if self._by_id is None or comment_id not in self._by_id.index:
            return
        self._current_id = comment_id
        row = self._by_id.loc[comment_id]
        clean = row["text_clean"] if "text_clean" in row.index else variant_text(row["text"], "clean")
        self._set_comment_text(
            f"{row['channel_handle']} | {row['title']} | {str(row['published_at'])[:10]} | polubienia: {row['like_count']}\n\n"
            f"Oryginał:\n{row['text']}\n\nPo czyszczeniu (text_clean):\n{clean}")
        self._predict_into(self.comment_panel, comment_id, str(row["text"]))

    def random_comment(self) -> None:
        if self._filtered.empty:
            return
        cid = random.choice(self._filtered["comment_id"].tolist())
        self.comments_tree.selection_set(cid)
        self.comments_tree.see(cid)
        self.show_comment(cid)

    def predict_custom(self) -> None:
        text = self.custom_text.get("1.0", "end").strip()
        clean = variant_text(text, "clean")
        self.clean_preview_var.set(f"Po czyszczeniu (text_clean): {clean}" if clean else "")
        self._predict_into(self.custom_panel, None, text)

    def refresh_summary(self) -> None:
        """Przycisk 'Odśwież': wczytuje predictions.csv od nowa (mogl go dopisac skrypt wsadowy)."""
        if self.comments is None:
            return
        if not self._predictions_from_file:
            self._compute_summary()
            return
        self.summary_info_var.set("Wczytywanie predykcji…")
        path = self.predictions_path
        self._run(lambda: self._prepare_predictions(load_predictions(path)), self._on_predictions_reloaded)

    def _on_predictions_reloaded(self, bundle: dict) -> None:
        self._set_predictions(bundle)
        self._update_status()
        self._compute_summary()

    def _compute_summary(self) -> None:
        if self.comments is None or self.predictions is None:
            return
        tree = self.summary_tree
        tree.delete(*tree.get_children())
        self.summary_figure.clear()
        if self.predictions.empty:
            self.summary_info_var.set("Brak data/predictions.csv - uruchom: python -m sentiment.predict")
            self.summary_canvas.draw_idle()
            return
        self.summary_info_var.set("Liczenie podsumowania…")
        preds, comments = self.predictions, self.comments
        self._run(lambda: channel_summary(preds, comments), self._render_summary)

    def _render_summary(self, summary: pd.DataFrame) -> None:
        tree = self.summary_tree
        tree.delete(*tree.get_children())
        for r in summary.itertuples(index=False):
            tree.insert("", "end", values=(
                r.channel_handle, MODEL_SPECS[r.model].display_name if r.model in MODEL_SPECS else r.model, r.variant,
                _fmt_int(int(r.n)), f"{r.negative:.0%}", f"{r.neutral:.0%}", f"{r.positive:.0%}"))
        self.summary_info_var.set(f"Ocenionych komentarzy: {_fmt_int(summary.groupby('channel_handle')['n'].max().sum())}"
                                  if not summary.empty else "Brak predykcji dla wczytanych komentarzy")
        self.summary_figure.clear()
        if not summary.empty:
            ax = self.summary_figure.add_subplot(111)
            names = [f"{r.channel_handle} · {MODEL_SPECS[r.model].display_name if r.model in MODEL_SPECS else r.model}"
                     f" · {r.variant}" for r in summary.itertuples(index=False)]
            left = [0.0] * len(summary)
            for label in LABELS:
                values = summary[label].astype(float).tolist()
                ax.barh(names, values, left=left, color=COLORS[label], label=LABELS_PL[label])
                left = [a + b for a, b in zip(left, values)]
            ax.set_xlim(0, 1)
            ax.invert_yaxis()
            ax.tick_params(labelsize=7)
            ax.legend(loc="lower right", fontsize=7, ncol=3)
            self.summary_figure.tight_layout()
        self.summary_canvas.draw_idle()


def main() -> None:
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()


if __name__ == "__main__":
    main()
