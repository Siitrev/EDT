"""Pobieranie komentarzy spod filmow z kanalow YouTube przez YouTube Data API v3.

Uzycie:
    python scraper/youtube_comments.py @WojciechCejrowski @wideoprezentacje
    python scraper/youtube_comments.py --export          # tylko eksport do CSV

Klucz API: zmienna YOUTUBE_API_KEY (np. w pliku .env w katalogu projektu).

Dane trafiaja do data/comments.db (SQLite). Skrypt mozna przerwac i uruchomic
ponownie: filmy juz pobrane sa pomijane. Po wyczerpaniu dziennego limitu API
(10 000 jednostek) skrypt konczy sie czysto; kolejnego dnia kontynuuje.

Koszt w jednostkach API: channels.list = 1, playlistItems.list = 1 (50 filmow),
commentThreads.list = 1 (100 watkow), comments.list = 1 (100 odpowiedzi).
"""
import argparse
import csv
import hashlib
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB_PATH = DATA / "comments.db"
CSV_PATH = DATA / "comments.csv"
API = "https://www.googleapis.com/youtube/v3"
# Uzywane, gdy nie podano kanalow. W PowerShellu "@nazwa" bez cudzyslowu znika
# (operator splattingu), wiec podawaj kanaly bez "@" albo w cudzyslowie.
DEFAULT_CHANNELS = ["@WojciechCejrowski", "@wideoprezentacje"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    channel_handle TEXT,
    channel_id TEXT,
    title TEXT,
    published_at TEXT,
    status TEXT DEFAULT 'pending',      -- pending | done | disabled | error
    comments_fetched INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS comments (
    comment_id TEXT PRIMARY KEY,
    video_id TEXT,
    channel_handle TEXT,
    parent_id TEXT,                     -- NULL dla komentarza glownego
    author_hash TEXT,                   -- skrot SHA-256 nazwy autora (bez danych osobowych)
    text TEXT,
    like_count INTEGER,
    reply_count INTEGER,
    published_at TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_comments_video ON comments(video_id);
CREATE INDEX IF NOT EXISTS idx_comments_parent ON comments(parent_id);
CREATE TABLE IF NOT EXISTS quota_usage (
    day TEXT,                           -- doba wg czasu pacyficznego (tak resetuje sie limit YouTube)
    channel_handle TEXT,
    units INTEGER,
    PRIMARY KEY (day, channel_handle)
);
"""


class QuotaExceeded(Exception):
    """API zglosilo wyczerpanie limitu."""


class BudgetExhausted(Exception):
    """Kanal wykorzystal przydzielona mu czesc dziennego limitu."""


class Budget:
    """Dzienny limit jednostek API rozliczany per kanal, zapamietywany w bazie.

    Dzieki temu kilka uruchomien tego samego dnia nie przekroczy przydzialu.
    """

    def __init__(self, db, daily_limit: int):
        self.db = db
        self.daily = daily_limit
        self.day = datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
        self.used = dict(db.execute("SELECT channel_handle, units FROM quota_usage WHERE day=?", (self.day,)).fetchall())
        self.channel = None
        self.cap = 0

    def total(self) -> int:
        return sum(self.used.values())

    def start(self, channel: str, cap: int):
        self.channel, self.cap = channel, cap

    def spend(self):
        u = self.used.get(self.channel, 0)
        if u >= self.cap or self.total() >= self.daily:
            raise BudgetExhausted()
        self.used[self.channel] = u + 1

    def save(self):
        self.db.executemany(
            "INSERT INTO quota_usage(day, channel_handle, units) VALUES (?,?,?) "
            "ON CONFLICT(day, channel_handle) DO UPDATE SET units=excluded.units",
            [(self.day, ch, u) for ch, u in self.used.items()],
        )


class CommentsDisabled(Exception):
    pass


class YouTube:
    def __init__(self, key: str, session=None, budget: "Budget | None" = None):
        self.key = key
        self.http = session or requests.Session()
        self.budget = budget
        self.units = 0

    def commit(self, db):
        if self.budget:
            self.budget.save()
        db.commit()

    def get(self, endpoint: str, **params):
        params["key"] = self.key
        for attempt in range(5):
            if self.budget:
                self.budget.spend()
            r = self.http.get(f"{API}/{endpoint}", params=params, timeout=30)
            self.units += 1
            if r.status_code == 200:
                return r.json()
            reason = ""
            try:
                errors = r.json().get("error", {}).get("errors", [])
                reason = errors[0].get("reason", "") if errors else ""
            except ValueError:
                pass
            if reason in ("quotaExceeded", "dailyLimitExceeded"):
                raise QuotaExceeded()
            if reason == "commentsDisabled":
                raise CommentsDisabled()
            if r.status_code in (403, 404) and reason in ("videoNotFound", "forbidden", "channelNotFound"):
                raise CommentsDisabled()
            if r.status_code in (500, 503) or reason in ("rateLimitExceeded", "backendError"):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Blad API {r.status_code} {reason}: {r.text[:300]}")
        raise RuntimeError(f"API nie odpowiada po 5 probach: {endpoint}")

    def uploads_playlist(self, handle: str):
        data = self.get("channels", part="contentDetails,snippet", forHandle=handle)
        items = data.get("items") or []
        if not items:
            raise RuntimeError(f"Nie znaleziono kanalu {handle}")
        ch = items[0]
        return ch["id"], ch["contentDetails"]["relatedPlaylists"]["uploads"]

    def videos(self, playlist_id: str):
        token = None
        while True:
            params = dict(part="snippet,contentDetails", playlistId=playlist_id, maxResults=50)
            if token:
                params["pageToken"] = token
            data = self.get("playlistItems", **params)
            for it in data.get("items", []):
                sn = it["snippet"]
                yield {
                    "video_id": it["contentDetails"]["videoId"],
                    "title": sn.get("title", ""),
                    "published_at": it["contentDetails"].get("videoPublishedAt") or sn.get("publishedAt"),
                }
            token = data.get("nextPageToken")
            if not token:
                return

    def comment_threads(self, video_id: str):
        token = None
        while True:
            params = dict(part="snippet,replies", videoId=video_id, maxResults=100, textFormat="plainText", order="time")
            if token:
                params["pageToken"] = token
            data = self.get("commentThreads", **params)
            yield from data.get("items", [])
            token = data.get("nextPageToken")
            if not token:
                return

    def replies(self, parent_id: str):
        token = None
        while True:
            params = dict(part="snippet", parentId=parent_id, maxResults=100, textFormat="plainText")
            if token:
                params["pageToken"] = token
            data = self.get("comments", **params)
            yield from data.get("items", [])
            token = data.get("nextPageToken")
            if not token:
                return


def author_hash(snippet: dict) -> str:
    raw = snippet.get("authorChannelId", {}).get("value") or snippet.get("authorDisplayName", "")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def comment_row(c: dict, video_id: str, handle: str, parent_id=None, reply_count=0):
    sn = c["snippet"]
    return (
        c["id"],
        video_id,
        handle,
        parent_id,
        author_hash(sn),
        sn.get("textOriginal") or sn.get("textDisplay", ""),
        sn.get("likeCount", 0),
        reply_count,
        sn.get("publishedAt"),
        sn.get("updatedAt"),
    )


def save_comments(db, rows):
    db.executemany("INSERT OR REPLACE INTO comments VALUES (?,?,?,?,?,?,?,?,?,?)", rows)


def scrape_video(yt: YouTube, db, video_id: str, handle: str) -> int:
    count = 0
    batch = []
    for th in yt.comment_threads(video_id):
        top = th["snippet"]["topLevelComment"]
        total_replies = th["snippet"].get("totalReplyCount", 0)
        batch.append(comment_row(top, video_id, handle, reply_count=total_replies))
        inline = th.get("replies", {}).get("comments", [])
        reps = inline if len(inline) >= total_replies else list(yt.replies(top["id"]))
        for rep in reps:
            batch.append(comment_row(rep, video_id, handle, parent_id=top["id"]))
        if len(batch) >= 500:
            save_comments(db, batch)
            yt.commit(db)
            count += len(batch)
            batch = []
    save_comments(db, batch)
    count += len(batch)
    return count


def scrape_channel(yt: YouTube, db, handle: str):
    channel_id, uploads = yt.uploads_playlist(handle)
    print(f"[{handle}] kanal {channel_id}, pobieram liste filmow...", flush=True)
    for v in yt.videos(uploads):
        db.execute(
            "INSERT OR IGNORE INTO videos(video_id, channel_handle, channel_id, title, published_at) VALUES (?,?,?,?,?)",
            (v["video_id"], handle, channel_id, v["title"], v["published_at"]),
        )
    yt.commit(db)
    pending = db.execute(
        "SELECT video_id, title FROM videos WHERE channel_handle=? AND status IN ('pending','error') ORDER BY published_at DESC",
        (handle,),
    ).fetchall()
    total = db.execute("SELECT COUNT(*) FROM videos WHERE channel_handle=?", (handle,)).fetchone()[0]
    print(f"[{handle}] filmow: {total}, juz pobranych (pomijam): {total - len(pending)}, "
          f"do pobrania: {len(pending)} - numeracja ponizej liczy tylko te do pobrania", flush=True)
    for i, (vid, title) in enumerate(pending, 1):
        try:
            n = scrape_video(yt, db, vid, handle)
            db.execute("UPDATE videos SET status='done', comments_fetched=? WHERE video_id=?", (n, vid))
            print(f"[{handle}] {i}/{len(pending)} {vid} +{n} kom. | {title[:60]} | jednostki API: {yt.units}", flush=True)
        except CommentsDisabled:
            db.execute("UPDATE videos SET status='disabled' WHERE video_id=?", (vid,))
            print(f"[{handle}] {i}/{len(pending)} {vid} komentarze wylaczone", flush=True)
        except (QuotaExceeded, BudgetExhausted):
            yt.commit(db)  # zapisane czesci zostaja, film zostaje 'pending' i wroci przy nastepnym uruchomieniu
            raise
        except Exception as e:  # pojedynczy film nie zatrzymuje calosci
            db.execute("UPDATE videos SET status='error' WHERE video_id=?", (vid,))
            print(f"[{handle}] {i}/{len(pending)} {vid} BLAD: {e}", flush=True)
        yt.commit(db)


def run_with_budget(yt: YouTube, db, budget: Budget, handles):
    """Dzieli dzienny limit po rowno miedzy kanaly.

    Runda 1: kazdy kanal dostaje limit/N jednostek.
    Runda 2: jesli ktorys kanal pobral juz wszystko, niewykorzystana reszta
    limitu idzie na kanaly, ktorym zostaly filmy (zeby limit sie nie marnowal).
    """
    share = budget.daily // len(handles)
    finished = set()
    for h in handles:
        budget.start(h, share)
        try:
            scrape_channel(yt, db, h)
            finished.add(h)
            print(f"[{h}] wszystko pobrane ({budget.used.get(h, 0)}/{share} jednostek)", flush=True)
        except BudgetExhausted:
            print(f"[{h}] wykorzystana dzienna czesc limitu ({share} jednostek)", flush=True)
    left = [h for h in handles if h not in finished]
    if finished and left and budget.total() < budget.daily:
        print(f"Reszta limitu ({budget.daily - budget.total()} jednostek) idzie na: {', '.join(left)}", flush=True)
        for h in left:
            budget.start(h, budget.daily)  # ograniczenie daje lacznie zuzyty limit
            try:
                scrape_channel(yt, db, h)
            except BudgetExhausted:
                print("Wykorzystany caly dzienny limit.", flush=True)
                return


def export_csv(db, path: Path = CSV_PATH):
    cur = db.execute(
        """SELECT c.comment_id, c.channel_handle, c.video_id, v.title, c.parent_id IS NOT NULL AS is_reply,
                  c.parent_id, c.author_hash, c.like_count, c.reply_count, c.published_at, c.text
           FROM comments c LEFT JOIN videos v ON v.video_id = c.video_id
           ORDER BY c.channel_handle, c.published_at"""
    )
    cols = [d[0] for d in cur.description]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        n = 0
        for row in cur:
            w.writerow(row)
            n += 1
    return n


def summary(db):
    for handle, vids, done, disabled, err in db.execute(
        """SELECT channel_handle, COUNT(*), SUM(status='done'), SUM(status='disabled'), SUM(status='error')
           FROM videos GROUP BY channel_handle"""
    ):
        n = db.execute("SELECT COUNT(*) FROM comments WHERE channel_handle=?", (handle,)).fetchone()[0]
        print(f"  {handle}: filmy {done}/{vids} pobrane, wylaczone {disabled}, bledy {err}, komentarzy {n}")


def normalize_handle(s: str) -> str:
    """'https://www.youtube.com/@Kanal/videos' / 'Kanal' / '@Kanal' -> '@Kanal'."""
    m = re.search(r"@([\w.\-]+)", s)
    return "@" + (m.group(1) if m else s.strip("/").split("/")[-1])


def open_db(path: Path = DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    return db


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):  # polskie znaki w konsoli/logu Windows
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Pobiera komentarze z kanalow YouTube (YouTube Data API v3).")
    ap.add_argument("channels", nargs="*",
                    help=f"kanaly, np. WojciechCejrowski (domyslnie: {' '.join(DEFAULT_CHANNELS)})")
    ap.add_argument("--export", action="store_true", help="tylko eksport bazy do data/comments.csv")
    ap.add_argument("--daily-limit", type=int, default=10_000, help="dzienny limit jednostek API (domyslnie 10000)")
    args = ap.parse_args(argv)

    db = open_db()
    if not args.export:
        channels = [c for c in args.channels if c.strip()] or DEFAULT_CHANNELS
        print("Kanaly:", ", ".join(normalize_handle(c) for c in channels), flush=True)
        load_dotenv(ROOT / ".env")
        key = os.environ.get("YOUTUBE_API_KEY")
        if not key:
            sys.exit("Brak YOUTUBE_API_KEY. Dopisz go do pliku .env w katalogu projektu.")
        budget = Budget(db, args.daily_limit)
        yt = YouTube(key, budget=budget)
        try:
            run_with_budget(yt, db, budget, [normalize_handle(h) for h in channels])
        except QuotaExceeded:
            print("\nAPI zglosilo wyczerpanie limitu. Uruchom ponownie jutro - skrypt kontynuuje od miejsca przerwania.")
        except KeyboardInterrupt:
            print("\nPrzerwano. Postep zapisany.")
        finally:
            yt.commit(db)
        print(f"Jednostki API dzis ({budget.day} czasu pacyficznego): " +
              ", ".join(f"{ch} {u}" for ch, u in budget.used.items()) + f" | razem {budget.total()}/{budget.daily}")
    n = export_csv(db)
    print(f"Wyeksportowano {n} komentarzy do {CSV_PATH}")
    summary(db)


if __name__ == "__main__":
    main()
