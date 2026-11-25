import time
import tkinter as tk

import pytest

from app import gui
from app.test_data import comments_df, fake_models, preds_df
from sentiment import models as m


@pytest.fixture
def app():
    try:
        a = gui.App(comments=comments_df(), predictions=preds_df(), models_loader=fake_models, sync=True)
    except tk.TclError:
        pytest.skip("brak wyświetlacza dla tkintera")
    a.withdraw()
    yield a
    a.destroy()


def rows(tree):
    return [tree.item(i, "values") for i in tree.get_children()]


def test_filter_and_select_shows_all_models_and_variants(app):
    app.query_var.set("Świetny")
    app.apply_filter()
    assert app.comments_tree.get_children() == ("c1",)
    app.show_comment("c1")
    shown = app.comment_text.get("1.0", "end")
    assert "Świetny film" in shown and "świetny film" in shown
    preds = rows(app.pred_tree)
    assert [(r[0], r[1]) for r in preds] == [(s.display_name, v) for s in m.MODEL_SPECS.values() for v in m.VARIANTS]
    assert [r[7] for r in preds] == ["plik", "plik", "plik", "na żywo", "na żywo", "na żywo"]
    assert app.agreement_var.get() == "raw: niezgodne | clean: niezgodne"


def test_channel_filter(app):
    app.channel_var.set("@B")
    app.apply_filter()
    assert app.comments_tree.get_children() == ("c3",)
    app.channel_var.set("Wszystkie")
    app.apply_filter()
    assert len(app.comments_tree.get_children()) == 3


def test_custom_text_live_prediction(app):
    app.custom_text.insert("1.0", "Brawo")
    app.predict_custom()
    assert len(rows(app.custom_pred_tree)) == 6
    assert app.custom_agreement_var.get() == "raw: zgodne – negative | clean: zgodne – negative"


def test_empty_custom_text_shows_warning(app):
    app.custom_text.insert("1.0", "   ")
    app.predict_custom()
    assert rows(app.custom_pred_tree) == []
    assert app.custom_agreement_var.get() == "Pusty komentarz – brak predykcji"


def test_summary_tab(app):
    app.refresh_summary()
    assert {r[0] for r in rows(app.summary_tree)} == {"@A", "@B"}
    assert {r[2] for r in rows(app.summary_tree)} == {"raw", "clean"}


# ---------------------------------------------------------------- tryb asynchroniczny i bledy

def make_app(**kwargs):
    try:
        a = gui.App(comments=comments_df(), **kwargs)
    except tk.TclError:
        pytest.skip("brak wyświetlacza dla tkintera")
    a.withdraw()
    return a


def pump(app, cond, timeout=5.0):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        app.update()
        time.sleep(0.02)
    app.update()
    return cond()


def test_poll_survives_exception_in_callback():
    app = make_app(predictions=preds_df(), models_loader=fake_models, sync=False)
    try:
        assert pump(app, lambda: app.comments is not None)
        # poczekaj, az ucichna zadania startowe (auto-wybor komentarza laduje modele i nadpisuje pasek stanu)
        assert pump(app, lambda: app.models is not None and app._queue.empty())
        pump(app, lambda: False, timeout=0.3)
        got = []

        def boom(_):
            raise RuntimeError("awaria callbacku")

        app._run(lambda: 1, boom)
        app._run(lambda: 2, got.append)
        assert pump(app, lambda: got == [2])
        assert "awaria callbacku" in app.status_var.get()
    finally:
        app.destroy()


def test_failed_model_load_can_be_retried():
    calls = []

    def flaky_loader():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("brak internetu")
        return fake_models()

    app = make_app(predictions=preds_df(), models_loader=flaky_loader, sync=True)
    try:
        # przy starcie automatycznie wybrany komentarz (c2, bez predykcji w pliku) probuje zaladowac modele
        assert len(calls) == 1
        assert app.agreement_var.get().startswith("Błąd ładowania modeli")
        assert rows(app.pred_tree) == []
        app.custom_text.insert("1.0", "Brawo")
        app.predict_custom()  # kolejna proba laduje modele od nowa zamiast czekac w nieskonczonosc
        assert len(calls) == 2
        assert len(rows(app.custom_pred_tree)) == 6
    finally:
        app.destroy()


def test_stale_prediction_not_drawn_after_empty_filter():
    def slow_loader():
        time.sleep(0.5)
        return fake_models()

    app = make_app(predictions=preds_df(), models_loader=slow_loader, sync=False)
    try:
        assert pump(app, lambda: app.comments is not None and app._current_id is not None)
        app.query_var.set("nie ma takiego tekstu")
        app.apply_filter()
        assert pump(app, lambda: app.comments_tree.get_children() == ())
        pump(app, lambda: app.models is not None)
        pump(app, lambda: False, timeout=0.5)
        assert rows(app.pred_tree) == []
    finally:
        app.destroy()


def test_refresh_reloads_predictions_file(tmp_path):
    path = tmp_path / "p.csv"
    preds_df().head(3).to_csv(path, index=False)
    app = make_app(predictions=None, predictions_path=path, models_loader=fake_models, sync=True)
    try:
        app.refresh_summary()
        assert {r[0] for r in rows(app.summary_tree)} == {"@A"}
        preds_df().to_csv(path, index=False)  # skrypt wsadowy dopisal nowe predykcje
        app.refresh_summary()
        assert {r[0] for r in rows(app.summary_tree)} == {"@A", "@B"}
        assert "predykcji w pliku: 4" in app.status_var.get()
    finally:
        app.destroy()


def test_cached_rows_uses_index(app):
    assert sorted(app._cached_rows("c1")["model"]) == ["herbert_twitter", "herbert_twitter", "polbert_politics"]
    assert app._cached_rows("c2").empty
