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

VALID_STATUSES = {
    "new",
    "triage",
    "assigned",
    "investigating",
    "awaiting_approval",
    "ack",
    "confirmed",
    "false_positive",
    "ignored",
    "closed",
}

STATUS_TRANSITIONS = {
    "new": {"triage", "assigned", "ack", "ignored", "closed"},
    "triage": {"assigned", "investigating", "confirmed", "false_positive", "ignored", "closed"},
    "assigned": {"investigating", "awaiting_approval", "confirmed", "false_positive", "ignored", "closed"},
    "investigating": {"awaiting_approval", "confirmed", "false_positive", "closed"},
    "awaiting_approval": {"confirmed", "false_positive", "closed"},
    "ack": {"triage", "assigned", "investigating", "confirmed", "false_positive", "ignored", "closed"},
    "confirmed": {"closed"},
    "false_positive": {"closed"},
    "ignored": {"closed"},
    "closed": set(),
}


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
        assignee    TEXT,
        approver    TEXT,
        root_cause_code TEXT,
        corrective_action_code TEXT,
        policy_clause TEXT,
        risk_score INTEGER,
        due_at      TEXT,
        closed_at   TEXT,
        merged_into INTEGER,
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

    CREATE TABLE IF NOT EXISTS incident_transitions (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_id   INTEGER NOT NULL,
        from_status TEXT,
        to_status  TEXT NOT NULL,
        actor      TEXT,
        note       TEXT,
        ts         TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(alert_id) REFERENCES alerts(id)
    );

    CREATE INDEX IF NOT EXISTS idx_transitions_alert ON incident_transitions(alert_id);

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
        "assignee": "ALTER TABLE alerts ADD COLUMN assignee TEXT",
        "approver": "ALTER TABLE alerts ADD COLUMN approver TEXT",
        "root_cause_code": "ALTER TABLE alerts ADD COLUMN root_cause_code TEXT",
        "corrective_action_code": "ALTER TABLE alerts ADD COLUMN corrective_action_code TEXT",
        "policy_clause": "ALTER TABLE alerts ADD COLUMN policy_clause TEXT",
        "risk_score": "ALTER TABLE alerts ADD COLUMN risk_score INTEGER",
        "due_at": "ALTER TABLE alerts ADD COLUMN due_at TEXT",
        "closed_at": "ALTER TABLE alerts ADD COLUMN closed_at TEXT",
        "merged_into": "ALTER TABLE alerts ADD COLUMN merged_into INTEGER",
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


def alert_get(aid: int) -> dict | None:
    con = db_conn()
    row = con.execute("SELECT * FROM alerts WHERE id=?", (aid,)).fetchone()
    con.close()
    return dict(row) if row else None


def incident_add_transition(alert_id: int, from_status: str | None,
                            to_status: str, actor: str, note: str = ""):
    con = db_conn()
    con.execute(
        "INSERT INTO incident_transitions(alert_id,from_status,to_status,actor,note) VALUES (?,?,?,?,?)",
        (alert_id, from_status, to_status, actor, note),
    )
    con.commit()
    con.close()


def incident_transition(aid: int, to_status: str, actor: str = "operator",
                        note: str = "", reviewer: str | None = None):
    if to_status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{to_status}', must be one of {VALID_STATUSES}")
    row = alert_get(aid)
    if not row:
        raise ValueError(f"Alert {aid} not found")

    cur = row.get("status") or "new"
    allowed = STATUS_TRANSITIONS.get(cur, set())
    if to_status != cur and to_status not in allowed:
        raise ValueError(f"Invalid transition {cur} -> {to_status}")

    closed_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") if to_status == "closed" else None
    con = db_conn()
    con.execute(
        """
        UPDATE alerts
           SET status=?, reviewer=?, reviewed_at=CURRENT_TIMESTAMP,
               closed_at=COALESCE(?, closed_at)
         WHERE id=?
        """,
        (to_status, reviewer or actor, closed_at, aid),
    )
    con.execute(
        "INSERT INTO incident_transitions(alert_id,from_status,to_status,actor,note) VALUES (?,?,?,?,?)",
        (aid, cur, to_status, actor, note),
    )
    con.commit()
    con.close()


def incident_assign(aid: int, assignee: str, actor: str = "operator",
                    due_at: str | None = None):
    row = alert_get(aid)
    if not row:
        raise ValueError(f"Alert {aid} not found")

    con = db_conn()
    con.execute(
        "UPDATE alerts SET assignee=?, due_at=COALESCE(?, due_at) WHERE id=?",
        (assignee, due_at, aid),
    )
    con.execute(
        "INSERT INTO incident_transitions(alert_id,from_status,to_status,actor,note) VALUES (?,?,?,?,?)",
        (aid, row.get("status"), row.get("status") or "new", actor, f"assigned:{assignee}"),
    )
    con.commit()
    con.close()


def incident_update_fields(aid: int, actor: str = "operator", **fields):
    allowed = {
        "type", "level", "root_cause_code", "corrective_action_code", "policy_clause",
        "risk_score", "due_at", "approver", "site", "label",
    }
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return

    cols = ", ".join([f"{k}=?" for k in updates.keys()])
    args = list(updates.values()) + [aid]
    con = db_conn()
    con.execute(f"UPDATE alerts SET {cols} WHERE id=?", args)
    con.execute(
        "INSERT INTO incident_transitions(alert_id,from_status,to_status,actor,note) VALUES (?,?,?,?,?)",
        (aid, None, (alert_get(aid) or {}).get("status", "new"), actor, "field_update"),
    )
    con.commit()
    con.close()


def incident_merge(master_id: int, child_id: int, actor: str = "operator"):
    if master_id == child_id:
        raise ValueError("master_id and child_id must be different")
    m = alert_get(master_id)
    c = alert_get(child_id)
    if not m or not c:
        raise ValueError("master or child incident not found")

    con = db_conn()
    con.execute("UPDATE alerts SET merged_into=? WHERE id=?", (master_id, child_id))
    con.execute(
        "INSERT INTO incident_transitions(alert_id,from_status,to_status,actor,note) VALUES (?,?,?,?,?)",
        (child_id, c.get("status"), c.get("status") or "new", actor, f"merged_into:{master_id}"),
    )
    con.commit()
    con.close()


def incident_queue(status: str | None = None, overdue: bool = False,
                   limit: int = 100, include_children: bool = False) -> list[dict]:
    sql = "SELECT * FROM alerts WHERE 1=1"
    args: list = []
    if status:
        sql += " AND status=?"
        args.append(status)
    if overdue:
        sql += " AND due_at IS NOT NULL AND due_at <> '' AND due_at < CURRENT_TIMESTAMP AND status <> 'closed'"
    if not include_children:
        sql += " AND (merged_into IS NULL OR merged_into = 0)"
    sql += " ORDER BY CASE level WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END, ts DESC LIMIT ?"
    args.append(max(1, min(limit, 500)))
    con = db_conn()
    rows = con.execute(sql, args).fetchall()
    con.close()
    return [dict(r) for r in rows]


def incident_timeline(aid: int) -> list[dict]:
    con = db_conn()
    rows = con.execute(
        "SELECT * FROM incident_transitions WHERE alert_id=? ORDER BY ts ASC, id ASC",
        (aid,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def ops_kpi(since: str | None = None) -> dict:
    con = db_conn()
    where = ""
    args: list = []
    if since:
        where = "WHERE ts >= ?"
        args.append(since)

    total = int(con.execute(f"SELECT COUNT(*) FROM alerts {where}", args).fetchone()[0])
    backlog = int(con.execute(
        f"SELECT COUNT(*) FROM alerts {where + (' AND ' if where else 'WHERE ')} status <> 'closed'",
        args,
    ).fetchone()[0])
    false_pos = int(con.execute(
        f"SELECT COUNT(*) FROM alerts {where + (' AND ' if where else 'WHERE ')} status = 'false_positive'",
        args,
    ).fetchone()[0])
    confirmed = int(con.execute(
        f"SELECT COUNT(*) FROM alerts {where + (' AND ' if where else 'WHERE ')} status = 'confirmed'",
        args,
    ).fetchone()[0])

    # MTTA: new -> first ack/triage/assigned/investigating/confirmed/false_positive/closed
    mtta_rows = con.execute(
        """
        SELECT a.id,
               MIN(CASE WHEN t.to_status IN ('ack','triage','assigned','investigating','confirmed','false_positive','closed')
                        THEN (strftime('%s', t.ts) - strftime('%s', a.ts)) END) AS sec
          FROM alerts a
          LEFT JOIN incident_transitions t ON t.alert_id = a.id
         GROUP BY a.id
        """
    ).fetchall()
    mtta_vals = [int(r["sec"]) for r in mtta_rows if r["sec"] is not None and r["sec"] >= 0]

    # MTTR: alert ts -> closed_at
    mttr_rows = con.execute(
        "SELECT (strftime('%s', closed_at) - strftime('%s', ts)) AS sec FROM alerts WHERE closed_at IS NOT NULL"
    ).fetchall()
    mttr_vals = [int(r["sec"]) for r in mttr_rows if r["sec"] is not None and r["sec"] >= 0]
    con.close()

    def _avg(vals: list[int]) -> float | None:
        if not vals:
            return None
        return round(sum(vals) / len(vals), 2)

    reviewed = confirmed + false_pos
    fp_rate = round(false_pos / reviewed, 4) if reviewed else None
    return {
        "total": total,
        "backlog": backlog,
        "confirmed": confirmed,
        "false_positive": false_pos,
        "false_positive_rate": fp_rate,
        "mtta_seconds": _avg(mtta_vals),
        "mttr_seconds": _avg(mttr_vals),
    }


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
