"""SQLite persistence. Everything is a time series so the engine can learn."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from .config import DB_PATH
from .models import (CategoryBid, GMPSnapshot, IPO, SubscriptionSnapshot,
                     Verdict)
from .util import now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS ipos (
    symbol TEXT PRIMARY KEY,
    name TEXT, status TEXT, board TEXT,
    open_date TEXT, close_date TEXT, listing_date TEXT,
    price_low REAL, price_high REAL, lot_size INTEGER,
    issue_size_shares REAL, issue_size_cr REAL,
    fresh_issue_cr REAL, ofs_cr REAL, face_value REAL,
    registrar TEXT, sector TEXT,
    meta TEXT, first_seen TEXT, last_seen TEXT
);

CREATE TABLE IF NOT EXISTS subscription (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, ts TEXT NOT NULL,
    category TEXT NOT NULL, offered REAL, bid REAL, times REAL,
    source TEXT DEFAULT 'nse'
);
CREATE INDEX IF NOT EXISTS ix_sub ON subscription(symbol, category, ts);

CREATE TABLE IF NOT EXISTS gmp (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, ts TEXT NOT NULL,
    gmp REAL, price REAL, est_listing REAL, est_gain_pct REAL,
    source TEXT, raw_name TEXT
);
CREATE INDEX IF NOT EXISTS ix_gmp ON gmp(symbol, ts);

CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT, ts TEXT, published TEXT,
    title TEXT, url TEXT UNIQUE, source TEXT,
    sentiment REAL, summary TEXT
);
CREATE INDEX IF NOT EXISTS ix_news ON news(symbol, ts);

CREATE TABLE IF NOT EXISTS financials (
    symbol TEXT, fy TEXT, payload TEXT, ts TEXT,
    PRIMARY KEY (symbol, fy)
);

CREATE TABLE IF NOT EXISTS verdicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, ts TEXT NOT NULL,
    score REAL, grade TEXT, payload TEXT
);
CREATE INDEX IF NOT EXISTS ix_verdict ON verdicts(symbol, ts);

-- every forecast we make, so we can score ourselves later
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, ts TEXT NOT NULL, horizon TEXT,
    metric TEXT, value REAL, lo REAL, hi REAL,
    features TEXT, model TEXT
);
CREATE INDEX IF NOT EXISTS ix_pred ON predictions(symbol, metric, ts);

-- ground truth once it lands
CREATE TABLE IF NOT EXISTS outcomes (
    symbol TEXT PRIMARY KEY,
    listing_date TEXT, listing_price REAL, listing_gain_pct REAL,
    close_day1 REAL, final_sub TEXT,
    actual_retail_p REAL, actual_retail_applications REAL,
    payload TEXT, ts TEXT
);

CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY, model TEXT, ts TEXT, response TEXT
);

-- macro snapshots, so rate CHANGES can be measured rather than just levels
CREATE TABLE IF NOT EXISTS macro (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, india_2y REAL, india_5y REAL, india_10y REAL,
    india_30y REAL, term_spread REAL, credit_spread REAL,
    credit_spread_source TEXT, index_change_pct REAL,
    curve_points INTEGER, payload TEXT
);
CREATE INDEX IF NOT EXISTS ix_macro ON macro(ts);

CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, ts TEXT);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, level TEXT, symbol TEXT, kind TEXT, message TEXT
);
"""


class Store:
    def __init__(self, path: Path | str = DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # -------------------------------------------------------------- IPOs
    def upsert_ipo(self, ipo: IPO) -> None:
        d = asdict(ipo)
        d["meta"] = json.dumps(d.get("meta") or {})
        ts = now_iso()
        with self.tx() as c:
            cur = c.execute("SELECT first_seen, meta FROM ipos WHERE symbol=?",
                            (ipo.symbol,))
            row = cur.fetchone()
            first_seen = row["first_seen"] if row else ts
            if row:  # merge meta so later sources enrich rather than clobber
                try:
                    old = json.loads(row["meta"] or "{}")
                    old.update(json.loads(d["meta"]))
                    d["meta"] = json.dumps(old)
                except Exception:
                    pass
            c.execute("""
                INSERT INTO ipos (symbol,name,status,board,open_date,close_date,
                    listing_date,price_low,price_high,lot_size,issue_size_shares,
                    issue_size_cr,fresh_issue_cr,ofs_cr,face_value,registrar,
                    sector,meta,first_seen,last_seen)
                VALUES (:symbol,:name,:status,:board,:open_date,:close_date,
                    :listing_date,:price_low,:price_high,:lot_size,
                    :issue_size_shares,:issue_size_cr,:fresh_issue_cr,:ofs_cr,
                    :face_value,:registrar,:sector,:meta,:first_seen,:last_seen)
                ON CONFLICT(symbol) DO UPDATE SET
                    name=excluded.name, status=excluded.status,
                    board=excluded.board, open_date=excluded.open_date,
                    close_date=excluded.close_date,
                    listing_date=COALESCE(excluded.listing_date, ipos.listing_date),
                    price_low=COALESCE(excluded.price_low, ipos.price_low),
                    price_high=COALESCE(excluded.price_high, ipos.price_high),
                    lot_size=COALESCE(excluded.lot_size, ipos.lot_size),
                    issue_size_shares=COALESCE(excluded.issue_size_shares, ipos.issue_size_shares),
                    issue_size_cr=COALESCE(excluded.issue_size_cr, ipos.issue_size_cr),
                    fresh_issue_cr=COALESCE(excluded.fresh_issue_cr, ipos.fresh_issue_cr),
                    ofs_cr=COALESCE(excluded.ofs_cr, ipos.ofs_cr),
                    face_value=COALESCE(excluded.face_value, ipos.face_value),
                    registrar=COALESCE(excluded.registrar, ipos.registrar),
                    sector=COALESCE(excluded.sector, ipos.sector),
                    meta=excluded.meta, last_seen=excluded.last_seen
            """, {**d, "first_seen": first_seen, "last_seen": ts})

    def get_ipos(self, statuses: tuple[str, ...] | None = None) -> list[IPO]:
        q = "SELECT * FROM ipos"
        args: tuple = ()
        if statuses:
            q += f" WHERE status IN ({','.join('?' * len(statuses))})"
            args = statuses
        q += " ORDER BY close_date IS NULL, close_date, symbol"
        rows = self._conn.execute(q, args).fetchall()
        return [self._row_to_ipo(r) for r in rows]

    def get_ipo(self, symbol: str) -> IPO | None:
        r = self._conn.execute("SELECT * FROM ipos WHERE symbol=?", (symbol,)).fetchone()
        return self._row_to_ipo(r) if r else None

    @staticmethod
    def _row_to_ipo(r: sqlite3.Row) -> IPO:
        d = dict(r)
        d.pop("first_seen", None)
        d.pop("last_seen", None)
        try:
            d["meta"] = json.loads(d.get("meta") or "{}")
        except Exception:
            d["meta"] = {}
        return IPO(**d)

    # ------------------------------------------------------ subscription
    def add_subscription(self, snap: SubscriptionSnapshot) -> None:
        with self.tx() as c:
            for cat, cb in snap.categories.items():
                c.execute(
                    "INSERT INTO subscription (symbol,ts,category,offered,bid,times,source)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (snap.symbol, snap.ts, cat, cb.offered, cb.bid, cb.times, snap.source))

    def latest_subscription(self, symbol: str) -> SubscriptionSnapshot | None:
        row = self._conn.execute(
            "SELECT ts FROM subscription WHERE symbol=? ORDER BY ts DESC LIMIT 1",
            (symbol,)).fetchone()
        if not row:
            return None
        ts = row["ts"]
        rows = self._conn.execute(
            "SELECT * FROM subscription WHERE symbol=? AND ts=?", (symbol, ts)).fetchall()
        cats = {r["category"]: CategoryBid(r["category"], r["offered"], r["bid"],
                                           r["times"]) for r in rows}
        return SubscriptionSnapshot(symbol, ts, cats)

    def subscription_series(self, symbol: str, category: str) -> list[tuple[str, float]]:
        rows = self._conn.execute(
            "SELECT ts, times FROM subscription WHERE symbol=? AND category=?"
            " AND times IS NOT NULL ORDER BY ts", (symbol, category)).fetchall()
        return [(r["ts"], r["times"]) for r in rows]

    # --------------------------------------------------------------- GMP
    def add_gmp(self, g: GMPSnapshot) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO gmp (symbol,ts,gmp,price,est_listing,"
                      "est_gain_pct,source,raw_name) VALUES (?,?,?,?,?,?,?,?)",
                      (g.symbol, g.ts, g.gmp, g.price, g.est_listing,
                       g.est_gain_pct, g.source, g.raw_name))

    def latest_gmp(self, symbol: str) -> GMPSnapshot | None:
        r = self._conn.execute(
            "SELECT * FROM gmp WHERE symbol=? ORDER BY ts DESC LIMIT 1",
            (symbol,)).fetchone()
        if not r:
            return None
        return GMPSnapshot(r["symbol"], r["ts"], r["gmp"], r["price"],
                           r["est_listing"], r["est_gain_pct"], r["source"],
                           r["raw_name"] or "")

    def gmp_series(self, symbol: str, limit: int = 60) -> list[tuple[str, float]]:
        rows = self._conn.execute(
            "SELECT ts, gmp FROM gmp WHERE symbol=? AND gmp IS NOT NULL"
            " ORDER BY ts DESC LIMIT ?", (symbol, limit)).fetchall()
        return [(r["ts"], r["gmp"]) for r in reversed(rows)]

    # -------------------------------------------------------------- news
    def add_news(self, symbol: str, published: str, title: str, url: str,
                 source: str, sentiment: float | None = None,
                 summary: str = "") -> bool:
        try:
            with self.tx() as c:
                c.execute("INSERT OR IGNORE INTO news (symbol,ts,published,title,"
                          "url,source,sentiment,summary) VALUES (?,?,?,?,?,?,?,?)",
                          (symbol, now_iso(), published, title, url, source,
                           sentiment, summary))
                return c.total_changes > 0
        except Exception:
            return False

    def recent_news(self, symbol: str, limit: int = 15) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM news WHERE symbol=? ORDER BY ts DESC LIMIT ?",
            (symbol, limit)).fetchall()
        return [dict(r) for r in rows]

    def news_needing_sentiment(self, limit: int = 25) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM news WHERE sentiment IS NULL ORDER BY ts DESC LIMIT ?",
            (limit,)).fetchall()
        return [dict(r) for r in rows]

    def set_sentiment(self, news_id: int, sentiment: float, summary: str = "") -> None:
        with self.tx() as c:
            c.execute("UPDATE news SET sentiment=?, summary=? WHERE id=?",
                      (sentiment, summary, news_id))

    # -------------------------------------------------------- financials
    def set_financials(self, symbol: str, fy: str, payload: dict[str, Any]) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO financials (symbol,fy,payload,ts)"
                      " VALUES (?,?,?,?)", (symbol, fy, json.dumps(payload), now_iso()))

    def get_financials(self, symbol: str) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT fy, payload FROM financials WHERE symbol=? ORDER BY fy",
            (symbol,)).fetchall()
        out = {}
        for r in rows:
            try:
                out[r["fy"]] = json.loads(r["payload"])
            except Exception:
                pass
        return out

    # ----------------------------------------------------------- verdicts
    def add_verdict(self, v: Verdict) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO verdicts (symbol,ts,score,grade,payload)"
                      " VALUES (?,?,?,?,?)",
                      (v.symbol, v.ts, v.score, v.grade, json.dumps(v.to_dict(),
                                                                   default=str)))

    def latest_verdict(self, symbol: str) -> dict[str, Any] | None:
        r = self._conn.execute(
            "SELECT payload FROM verdicts WHERE symbol=? ORDER BY ts DESC LIMIT 1",
            (symbol,)).fetchone()
        if not r:
            return None
        try:
            return json.loads(r["payload"])
        except Exception:
            return None

    # -------------------------------------------------- predictions/truth
    def add_prediction(self, symbol: str, metric: str, value: float,
                       lo: float | None = None, hi: float | None = None,
                       features: dict[str, Any] | None = None,
                       model: str = "quant", horizon: str = "listing") -> None:
        with self.tx() as c:
            c.execute("INSERT INTO predictions (symbol,ts,horizon,metric,value,"
                      "lo,hi,features,model) VALUES (?,?,?,?,?,?,?,?,?)",
                      (symbol, now_iso(), horizon, metric, value, lo, hi,
                       json.dumps(features or {}, default=str), model))

    def set_outcome(self, symbol: str, **kw: Any) -> None:
        cols = ("listing_date", "listing_price", "listing_gain_pct", "close_day1",
                "final_sub", "actual_retail_p", "actual_retail_applications",
                "payload")
        vals = {c: kw.get(c) for c in cols}
        for k in ("final_sub", "payload"):
            if isinstance(vals[k], (dict, list)):
                vals[k] = json.dumps(vals[k], default=str)
        with self.tx() as c:
            c.execute(f"INSERT OR REPLACE INTO outcomes (symbol,{','.join(cols)},ts)"
                      f" VALUES (?,{','.join('?' * len(cols))},?)",
                      (symbol, *[vals[c] for c in cols], now_iso()))

    def get_outcome(self, symbol: str) -> dict[str, Any] | None:
        r = self._conn.execute("SELECT * FROM outcomes WHERE symbol=?",
                               (symbol,)).fetchone()
        return dict(r) if r else None

    def training_rows(self) -> list[dict[str, Any]]:
        """Joined (features, truth) pairs for the self-calibration pass."""
        rows = self._conn.execute("""
            SELECT p.symbol, p.metric, p.value AS predicted, p.features,
                   o.listing_gain_pct, o.actual_retail_p, o.final_sub
            FROM predictions p JOIN outcomes o ON o.symbol = p.symbol
            WHERE o.listing_gain_pct IS NOT NULL
        """).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["features"] = json.loads(d["features"] or "{}")
            except Exception:
                d["features"] = {}
            out.append(d)
        return out

    # ------------------------------------------------------------- macro
    def add_macro(self, snap: Any) -> None:
        d = snap.to_dict() if hasattr(snap, "to_dict") else dict(snap)
        with self.tx() as c:
            c.execute("INSERT INTO macro (ts,india_2y,india_5y,india_10y,"
                      "india_30y,term_spread,credit_spread,credit_spread_source,"
                      "index_change_pct,curve_points,payload)"
                      " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (d.get("ts"), d.get("india_2y"), d.get("india_5y"),
                       d.get("india_10y"), d.get("india_30y"),
                       d.get("term_spread"), d.get("credit_spread"),
                       d.get("credit_spread_source"), d.get("index_change_pct"),
                       d.get("curve_points"), json.dumps(d, default=str)))

    def macro_history(self, limit: int = 90) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM macro ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def latest_macro(self) -> dict[str, Any] | None:
        r = self._conn.execute(
            "SELECT * FROM macro ORDER BY ts DESC LIMIT 1").fetchone()
        return dict(r) if r else None

    # ------------------------------------------------------------- misc
    def cache_llm(self, key: str, model: str, response: str) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO llm_cache (key,model,ts,response)"
                      " VALUES (?,?,?,?)", (key, model, now_iso(), response))

    def get_llm_cache(self, key: str, max_age_s: float | None = None) -> str | None:
        r = self._conn.execute("SELECT ts, response FROM llm_cache WHERE key=?",
                               (key,)).fetchone()
        if not r:
            return None
        if max_age_s is not None:
            from datetime import datetime
            try:
                age = (datetime.now() - datetime.fromisoformat(r["ts"])).total_seconds()
                if age > max_age_s:
                    return None
            except Exception:
                pass
        return r["response"]

    def set_kv(self, key: str, value: Any) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO kv (key,value,ts) VALUES (?,?,?)",
                      (key, json.dumps(value, default=str), now_iso()))

    def get_kv(self, key: str, default: Any = None) -> Any:
        r = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        if not r:
            return default
        try:
            return json.loads(r["value"])
        except Exception:
            return default

    def log_event(self, kind: str, message: str, symbol: str = "",
                  level: str = "INFO") -> None:
        with self.tx() as c:
            c.execute("INSERT INTO events (ts,level,symbol,kind,message)"
                      " VALUES (?,?,?,?,?)", (now_iso(), level, symbol, kind, message))

    def recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
