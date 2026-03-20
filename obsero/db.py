"""
obsero.db — SQLite database layer (WAL mode, thread-safe connections).

Extended schema:
  - alerts table gains: model_key, gpu_id, rule_json, full_image
  - statuses normalised: new, ack, confirmed, false_positive, ignored, closed
"""

from __future__ import annotations
import json, sqlite3, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "panel.db"

VALID_STATUSES = {"new", "ack", "confirmed", "false_positive", "ignored", "closed"}


def db_conn(path: Path | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(str(path or DB_PATH), timeout=10)
    con.row_factory = sqlite3.Row
    return con


def db_init(path: Path | None = None):
    """Create / migrate tables.  Safe to call on every start."""
    con = db_conn(path)
    cur = con.cursor()
    cur.executescript("""
    PRAGMA journal_mode=WAL;

    CREATE TABLE IF NOT EXISTS cameras (
        id   INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        code TEXT UNIQUE,
        url  TEXT NOT NULL,
        online        INTEGER DEFAULT 0,
        ptz_protocol  TEXT DEFAULT 'onvif',
        created_at    TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS alerts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          TEXT NOT NULL,
        camera_id   INTEGER,
        level       TEXT DEFAULT 'medium',
        type        TEXT,
        label       TEXT,
        conf        REAL,
        bbox        TEXT,
        image       TEXT,
        full_image  TEXT,
        model_key   TEXT,
        gpu_id      INTEGER,
        rule_json   TEXT,
        site        TEXT,
        status      TEXT DEFAULT 'new',
        reviewer    TEXT,
        reviewed_at TEXT,
        FOREIGN KEY(camera_id) REFERENCES cameras(id)
    );

    CREATE INDEX IF NOT EXISTS idx_alerts_ts     ON alerts(ts);
    CREATE INDEX IF NOT EXISTS idx_alerts_cam    ON alerts(camera_id);
    CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);

    CREATE TABLE IF NOT EXISTS users (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        role     TEXT
    );

    CREATE TABLE IF NOT EXISTS audit_logs (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        ts      TEXT DEFAULT CURRENT_TIMESTAMP,
        actor   TEXT,
        action  TEXT,
        details TEXT
    );

    CREATE TABLE IF NOT EXISTS patrol_tasks (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL,
        camera_ids  TEXT NOT NULL,
        patrol_type TEXT,
        frequency   TEXT,
        next_run    TEXT,
        last_report TEXT,
        active      INTEGER DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS alarm_levels (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT UNIQUE,
        level      TEXT
    );
    """)
    con.commit()

    # ---- migration: add missing columns to alerts (safe if already exist) ----
    _existing = {row[1] for row in cur.execute("PRAGMA table_info(alerts)").fetchall()}
    migrations = {
        "full_image": "ALTER TABLE alerts ADD COLUMN full_image TEXT",
        "model_key":  "ALTER TABLE alerts ADD COLUMN model_key TEXT",
        "gpu_id":     "ALTER TABLE alerts ADD COLUMN gpu_id INTEGER",
        "rule_json":  "ALTER TABLE alerts ADD COLUMN rule_json TEXT",
    }
    for col, ddl in migrations.items():
        if col not in _existing:
            try:
                cur.execute(ddl)
            except sqlite3.OperationalError:
                pass
    con.commit()
    con.close()


# ═══════════════════ Audit ════════════════════════════════════════════════════

def audit(actor: str, action: str, details: str = ""):
    con = db_conn()
    con.execute("INSERT INTO audit_logs(actor,action,details) VALUES (?,?,?)",
                (actor, action, details))
    con.commit()
    con.close()


# ═══════════════════ Cameras ═════════════════════════════════════════════════

def cameras_all() -> list[dict]:
    con = db_conn()
    rows = con.execute("SELECT * FROM cameras ORDER BY id").fetchall()
    con.close()
    return [dict(r) for r in rows]


def camera_by_id(cid: int) -> dict | None:
    con = db_conn()
    r = con.execute("SELECT * FROM cameras WHERE id=?", (cid,)).fetchone()
    con.close()
    return dict(r) if r else None


def camera_upsert(name: str, url: str, code: str | None = None,
                  ptz_protocol: str = "onvif", cid: int | None = None,
                  online: bool | None = None):
    con = db_conn()
    try:
        if cid is None:
            if code is not None:
                con.execute("""
                    INSERT INTO cameras (name, code, url, ptz_protocol, online)
                    VALUES (?,?,?,?,?)
                    ON CONFLICT(code) DO UPDATE SET
                        name = excluded.name,
                        url  = excluded.url,
                        ptz_protocol = excluded.ptz_protocol,
                        online = COALESCE(excluded.online, cameras.online)
                """, (name, code, url, ptz_protocol,
                      int(bool(online)) if online is not None else 0))
            else:
                con.execute(
                    "INSERT INTO cameras(name, code, url, ptz_protocol, online) VALUES (?,?,?,?,?)",
                    (name, code, url, ptz_protocol,
                     int(bool(online)) if online is not None else 0))
        else:
            if online is None:
                con.execute(
                    "UPDATE cameras SET name=?, code=?, url=?, ptz_protocol=? WHERE id=?",
                    (name, code, url, ptz_protocol, cid))
            else:
                con.execute(
                    "UPDATE cameras SET name=?, code=?, url=?, ptz_protocol=?, online=? WHERE id=?",
                    (name, code, url, ptz_protocol, int(bool(online)), cid))
        con.commit()
    finally:
        con.close()


def camera_set_online(cid: int, online: bool):
    con = db_conn()
    con.execute("UPDATE cameras SET online=? WHERE id=?", (1 if online else 0, cid))
    con.commit()
    con.close()


# ═══════════════════ Alerts ══════════════════════════════════════════════════

def alert_insert(ts: str, camera_id: int | None, level: str, etype: str,
                 label: str, conf: float, bbox: list, image: str,
                 site: str | None = None, status: str = "new",
                 model_key: str | None = None, gpu_id: int | None = None,
                 rule_json: str | None = None, full_image: str | None = None):
    con = db_conn()
    con.execute("""
        INSERT INTO alerts(ts, camera_id, level, type, label, conf, bbox, image,
                           full_image, model_key, gpu_id, rule_json, site, status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (ts, camera_id, level, etype, label, conf, json.dumps(bbox), image,
         full_image, model_key, gpu_id, rule_json, site, status))
    con.commit()
    con.close()


def alert_query(limit: int = 50, **filters) -> list[dict]:
    sql = "SELECT * FROM alerts WHERE 1=1"
    args: list = []
    if filters.get("camera_id"):
        sql += " AND camera_id=?"; args.append(int(filters["camera_id"]))
    if filters.get("level"):
        sql += " AND level=?"; args.append(filters["level"])
    if filters.get("type"):
        sql += " AND type=?"; args.append(filters["type"])
    if filters.get("t_from"):
        sql += " AND ts>=?"; args.append(filters["t_from"])
    if filters.get("t_to"):
        sql += " AND ts<=?"; args.append(filters["t_to"])
    if filters.get("status"):
        sql += " AND status=?"; args.append(filters["status"])
    sql += " ORDER BY ts DESC LIMIT ?"
    args.append(max(1, min(limit, 500)))
    con = db_conn()
    rows = con.execute(sql, args).fetchall()
    con.close()
    return [dict(r) for r in rows]


def alert_update_status(aid: int, status: str, reviewer: str | None = None):
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}', must be one of {VALID_STATUSES}")
    con = db_conn()
    con.execute(
        "UPDATE alerts SET status=?, reviewer=?, reviewed_at=CURRENT_TIMESTAMP WHERE id=?",
        (status, reviewer, aid))
    con.commit()
    con.close()


def precision_query(event_type: str | None = None,
                    since: str | None = None,
                    min_reviewed: int = 1) -> dict:
    """Compute precision = confirmed / (confirmed + false_positive)."""
    base = "SELECT status, COUNT(*) as cnt FROM alerts WHERE status IN ('confirmed','false_positive')"
    args: list = []
    if event_type:
        base += " AND type=?"; args.append(event_type)
    if since:
        base += " AND ts>=?"; args.append(since)
    base += " GROUP BY status"
    con = db_conn()
    rows = con.execute(base, args).fetchall()
    con.close()
    counts = {r["status"]: r["cnt"] for r in rows}
    confirmed = counts.get("confirmed", 0)
    false_pos = counts.get("false_positive", 0)
    total = confirmed + false_pos
    precision = round(confirmed / total, 4) if total >= min_reviewed else None
    return {
        "confirmed": confirmed,
        "false_positive": false_pos,
        "total_reviewed": total,
        "precision": precision,
        "min_reviewed_met": total >= min_reviewed,
    }
