import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import youtube_comments as yc  # noqa: E402


class Resp:
    def __init__(self, code, data):
        self.status_code = code
        self._d = data
        self.text = str(data)

    def json(self):
        return self._d


def c(cid, text, author="a"):
    return {"id": cid, "snippet": {"textOriginal": text, "authorDisplayName": author, "likeCount": 1,
                                   "publishedAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z"}}


class FakeSession:
    def __init__(self, quota_after=None):
        self.calls = 0
        self.quota_after = quota_after

    def get(self, url, params, timeout):
        self.calls += 1
        if self.quota_after is not None and self.calls > self.quota_after:
            return Resp(403, {"error": {"errors": [{"reason": "quotaExceeded"}]}})
        ep = url.rsplit("/", 1)[-1]
        tok = params.get("pageToken")
        if ep == "channels":
            h = params["forHandle"]
            return Resp(200, {"items": [{"id": "UC" + h, "contentDetails": {"relatedPlaylists": {"uploads": "UU" + h}}}]})
        if ep == "playlistItems":
            h = params["playlistId"][2:]
            if h == "@S":  # maly kanal bez filmow
                return Resp(200, {"items": []})
            vids = [f"{h}-v1", f"{h}-v2"] if not tok else [f"{h}-v3"]
            data = {"items": [{"snippet": {"title": "T " + v}, "contentDetails": {"videoId": v, "videoPublishedAt": "2026-01-0%dT00:00:00Z" % i}}
                              for i, v in enumerate(vids, 1)]}
            if not tok:
                data["nextPageToken"] = "p2"
            return Resp(200, data)
        if ep == "commentThreads":
            v = params["videoId"]
            if v.endswith("v3"):
                return Resp(403, {"error": {"errors": [{"reason": "commentsDisabled"}]}})
            if not tok:
                th = {"id": v + "t1", "snippet": {"topLevelComment": c(v + "t1", "super film"), "totalReplyCount": 1},
                      "replies": {"comments": [c(v + "t1.r1", "zgadzam sie")]}}
                return Resp(200, {"items": [th], "nextPageToken": "x"})
            th = {"id": v + "t2", "snippet": {"topLevelComment": c(v + "t2", "slabe"), "totalReplyCount": 7},
                  "replies": {"comments": [c(v + "t2.r1", "r")]}}
            return Resp(200, {"items": [th]})
        if ep == "comments":
            p = params["parentId"]
            return Resp(200, {"items": [c(f"{p}.r{i}", "odp") for i in range(7)]})
        raise AssertionError(ep)


def test_normalize_handle():
    assert yc.normalize_handle("https://www.youtube.com/@WojciechCejrowski/videos") == "@WojciechCejrowski"
    assert yc.normalize_handle("@wideoprezentacje") == "@wideoprezentacje"
    assert yc.normalize_handle("wideoprezentacje") == "@wideoprezentacje"


def test_full_scrape_labels_channel_and_fetches_replies(tmp_path):
    db = yc.open_db(tmp_path / "t.db")
    yt = yc.YouTube("k", FakeSession())
    for h in ("@A", "@B"):
        yc.scrape_channel(yt, db, h)
    db.commit()
    # na film: t1 + 1 odp. inline, t2 + 7 odp. dociagnietych osobno = 10; 2 filmy na kanal
    for h in ("@A", "@B"):
        n = db.execute("SELECT COUNT(*) FROM comments WHERE channel_handle=?", (h,)).fetchone()[0]
        assert n == 20, n
        st = dict(db.execute("SELECT video_id, status FROM videos WHERE channel_handle=?", (h,)).fetchall())
        assert st == {f"{h}-v1": "done", f"{h}-v2": "done", f"{h}-v3": "disabled"}
    # nick zahaszowany, nie jawny
    assert db.execute("SELECT COUNT(*) FROM comments WHERE author_hash='a'").fetchone()[0] == 0
    out = tmp_path / "c.csv"
    assert yc.export_csv(db, out) == 40
    rows = list(csv.DictReader(open(out, encoding="utf-8-sig")))
    assert {r["channel_handle"] for r in rows} == {"@A", "@B"}
    assert rows[0]["title"].startswith("T ")


def run(db, daily, handles):
    budget = yc.Budget(db, daily)
    yt = yc.YouTube("k", FakeSession(), budget=budget)
    yc.run_with_budget(yt, db, budget, handles)
    yt.commit(db)
    return budget


def test_budget_split_half_per_channel(tmp_path):
    db = yc.open_db(tmp_path / "t.db")
    b = run(db, 10, ["@A", "@B"])  # pelne pobranie kanalu kosztuje 10 jednostek
    assert b.used == {"@A": 5, "@B": 5}
    # kolejne uruchomienie tego samego dnia nic nie pobiera - przydzial juz zuzyty
    b2 = run(db, 10, ["@A", "@B"])
    assert b2.used == {"@A": 5, "@B": 5}


def test_budget_leftover_goes_to_unfinished_channel(tmp_path):
    db = yc.open_db(tmp_path / "t.db")
    b = run(db, 17, ["@S", "@A"])
    assert b.used["@S"] == 2 and b.total() <= 17
    st = {r[0] for r in db.execute("SELECT status FROM videos WHERE channel_handle='@A'")}
    assert st == {"done", "disabled"}


def test_quota_stops_and_resume_continues(tmp_path):
    db = yc.open_db(tmp_path / "t.db")
    yt = yc.YouTube("k", FakeSession(quota_after=5))
    try:
        yc.scrape_channel(yt, db, "@A")
        raise AssertionError("powinien byc QuotaExceeded")
    except yc.QuotaExceeded:
        pass
    db.commit()
    done_before = db.execute("SELECT COUNT(*) FROM videos WHERE status='done'").fetchone()[0]
    yc.scrape_channel(yc.YouTube("k", FakeSession()), db, "@A")
    db.commit()
    st = [r[0] for r in db.execute("SELECT status FROM videos ORDER BY video_id")]
    assert st == ["done", "done", "disabled"], (done_before, st)
    assert db.execute("SELECT COUNT(*) FROM comments").fetchone()[0] == 20
