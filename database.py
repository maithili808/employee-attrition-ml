"""
database.py  —  SQLite persistence layer for AttritionIQ
Tables:
  - predictions   : every prediction run (core facts + SHAP top reason)
  - shap_factors  : per-feature SHAP breakdown linked to each prediction
  - users         : login credentials + role
"""

import sqlite3
import hashlib
import json
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "attrition.db"


# ─────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row          # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL") # safe concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Schema creation  (run once at startup)
# ─────────────────────────────────────────────
def init_db():
    with get_db() as conn:
        conn.executescript("""
        -- ── Users ─────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            role          TEXT    NOT NULL DEFAULT 'Analyst',
            initials      TEXT    NOT NULL DEFAULT 'U',
            created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        -- ── Predictions ───────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS predictions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT    NOT NULL DEFAULT (datetime('now')),

            -- identity
            employee_id         TEXT,
            run_by              TEXT    NOT NULL,

            -- job context
            department          TEXT,
            job_role            TEXT,
            overtime            TEXT,

            -- raw inputs (snapshot for audit)
            age                 INTEGER,
            monthly_income      INTEGER,
            job_satisfaction    INTEGER,
            work_life_balance   INTEGER,
            years_at_company    INTEGER,
            distance_from_home  INTEGER,
            total_working_years INTEGER,
            job_level           INTEGER,
            marital_status      TEXT,
            business_travel     TEXT,

            -- model outputs
            probability         REAL    NOT NULL,
            risk_level          TEXT    NOT NULL,   -- High / Medium / Low
            prediction          TEXT    NOT NULL,   -- Yes / No label
            model_threshold     REAL,

            -- SHAP summary
            top_shap_feature    TEXT,
            top_shap_value      REAL,
            shap_json           TEXT    -- full top-12 as JSON string
        );

        -- ── SHAP factors (normalised rows) ────────────────────────────
        CREATE TABLE IF NOT EXISTS shap_factors (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id   INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
            rank            INTEGER NOT NULL,
            feature         TEXT    NOT NULL,
            shap_value      REAL    NOT NULL,
            direction       TEXT    NOT NULL,   -- 'risk' | 'safe'
            bar_pct         REAL
        );

        -- ── Indexes ───────────────────────────────────────────────────
        CREATE INDEX IF NOT EXISTS idx_pred_timestamp  ON predictions(timestamp);
        CREATE INDEX IF NOT EXISTS idx_pred_risk       ON predictions(risk_level);
        CREATE INDEX IF NOT EXISTS idx_pred_dept       ON predictions(department);
        CREATE INDEX IF NOT EXISTS idx_pred_run_by     ON predictions(run_by);
        CREATE INDEX IF NOT EXISTS idx_shap_pred_id    ON shap_factors(prediction_id);
        """)

        # Seed default users if table is empty
        cur = conn.execute("SELECT COUNT(*) FROM users")
        if cur.fetchone()[0] == 0:
            _seed_users(conn)


def _seed_users(conn):
    defaults = [
        ("hr@company.com",       "password", "HR Manager", "HR"),
        ("analyst@company.com",  "password", "Analyst",    "AN"),
        ("admin@company.com",    "password", "Admin",      "AD"),
    ]
    for email, pwd, role, initials in defaults:
        conn.execute(
            "INSERT INTO users (email, password_hash, role, initials) VALUES (?,?,?,?)",
            (email, _hash(pwd), role, initials)
        )


# ─────────────────────────────────────────────
# User helpers
# ─────────────────────────────────────────────
def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def get_user(email: str):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
    return dict(row) if row else None

def verify_user(email: str, password: str):
    user = get_user(email)
    if user and user["password_hash"] == _hash(password):
        return user
    return None


# ─────────────────────────────────────────────
# Save a prediction + SHAP rows
# ─────────────────────────────────────────────
def save_prediction(pred_dict: dict, shap_data: list, threshold: float) -> int:
    """
    Insert one prediction row + its SHAP factor rows.
    Returns the new prediction id.
    """
    top = shap_data[0] if shap_data else {}

    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO predictions (
                timestamp, employee_id, run_by,
                department, job_role, overtime,
                age, monthly_income, job_satisfaction, work_life_balance,
                years_at_company, distance_from_home, total_working_years,
                job_level, marital_status, business_travel,
                probability, risk_level, prediction, model_threshold,
                top_shap_feature, top_shap_value, shap_json
            ) VALUES (
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?
            )
        """, (
            pred_dict.get("timestamp",   datetime.now().strftime("%Y-%m-%d %H:%M")),
            pred_dict.get("employee_id"),
            pred_dict.get("run_by", "unknown"),

            pred_dict.get("department"),
            pred_dict.get("job_role"),
            pred_dict.get("overtime"),

            pred_dict.get("age"),
            pred_dict.get("monthly_income"),
            pred_dict.get("job_satisfaction"),
            pred_dict.get("work_life_balance"),
            pred_dict.get("years_at_company"),
            pred_dict.get("distance_from_home"),
            pred_dict.get("total_working_years"),
            pred_dict.get("job_level"),
            pred_dict.get("marital_status"),
            pred_dict.get("business_travel"),

            pred_dict["probability"],
            pred_dict["risk_level"],
            pred_dict["prediction"],
            threshold,

            top.get("feature"),
            top.get("shap_value"),
            json.dumps(shap_data),
        ))
        pred_id = cur.lastrowid

        # Insert SHAP factor rows
        for rank, s in enumerate(shap_data, start=1):
            conn.execute("""
                INSERT INTO shap_factors
                    (prediction_id, rank, feature, shap_value, direction, bar_pct)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (pred_id, rank, s["feature"], s["shap_value"],
                  s["direction"], s.get("bar_pct", 0)))

    return pred_id


# ─────────────────────────────────────────────
# Read helpers
# ─────────────────────────────────────────────
def get_predictions(risk_filter="all", limit=200, offset=0):
    """Fetch predictions newest-first, optional risk filter."""
    with get_db() as conn:
        if risk_filter == "all":
            rows = conn.execute("""
                SELECT * FROM predictions
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, (limit, offset)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM predictions
                WHERE LOWER(risk_level) = LOWER(?)
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, (risk_filter, limit, offset)).fetchall()
    return [dict(r) for r in rows]


def get_prediction_by_id(pred_id: int):
    with get_db() as conn:
        row  = conn.execute(
            "SELECT * FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
        shap = conn.execute(
            "SELECT * FROM shap_factors WHERE prediction_id = ? ORDER BY rank",
            (pred_id,)
        ).fetchall()
    if not row:
        return None, []
    return dict(row), [dict(s) for s in shap]


def count_predictions(risk_filter="all"):
    with get_db() as conn:
        if risk_filter == "all":
            return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        return conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE LOWER(risk_level)=LOWER(?)",
            (risk_filter,)
        ).fetchone()[0]


# ─────────────────────────────────────────────
# Dashboard aggregate stats
# ─────────────────────────────────────────────
def get_dashboard_stats():
    """All counts + totals the dashboard needs in one DB round-trip."""
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        high  = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_level='High'").fetchone()[0]
        med   = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_level='Medium'").fetchone()[0]
        low   = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_level='Low'").fetchone()[0]
        leave = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction LIKE 'Yes%'"
        ).fetchone()[0]

        # Recent 5
        recent = conn.execute("""
            SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 5
        """).fetchall()

        # Department distribution
        dept_rows = conn.execute("""
            SELECT department, COUNT(*) as cnt
            FROM predictions
            WHERE department IS NOT NULL
            GROUP BY department
            ORDER BY cnt DESC
        """).fetchall()

        # Risk over time (last 30 predictions, grouped by date)
        trend_rows = conn.execute("""
            SELECT DATE(timestamp) as day,
                   COUNT(*) as total,
                   SUM(CASE WHEN risk_level='High' THEN 1 ELSE 0 END) as high_count,
                   ROUND(AVG(probability),1) as avg_prob
            FROM predictions
            GROUP BY DATE(timestamp)
            ORDER BY day DESC
            LIMIT 30
        """).fetchall()

        # Top SHAP features across all predictions
        top_shap = conn.execute("""
            SELECT feature, COUNT(*) as freq,
                   ROUND(AVG(ABS(shap_value)),4) as avg_abs_shap
            FROM shap_factors
            WHERE rank <= 3
            GROUP BY feature
            ORDER BY freq DESC
            LIMIT 8
        """).fetchall()

        # Overtime breakdown
        ot_rows = conn.execute("""
            SELECT overtime,
                   COUNT(*) as cnt,
                   ROUND(AVG(probability),1) as avg_prob
            FROM predictions
            WHERE overtime IS NOT NULL
            GROUP BY overtime
        """).fetchall()

    return {
        "total":   total,
        "high":    high,
        "medium":  med,
        "low":     low,
        "leave":   leave,
        "stay":    total - leave,
        "recent":  [dict(r) for r in recent],
        "by_dept": [dict(r) for r in dept_rows],
        "trend":   [dict(r) for r in reversed(trend_rows)],
        "top_shap":[dict(r) for r in top_shap],
        "overtime":[dict(r) for r in ot_rows],
    }


def get_chart_data():
    """JSON-serialisable dict for /api/chart-data."""
    stats = get_dashboard_stats()
    return {
        "risk_dist": {
            "High":   stats["high"],
            "Medium": stats["medium"],
            "Low":    stats["low"],
        },
        "departments": {r["department"]: r["cnt"] for r in stats["by_dept"]},
        "trend": stats["trend"],
        "top_shap": stats["top_shap"],
        "overtime": stats["overtime"],
    }
# =========================================================
# 🔥 COMPATIBILITY WRAPPERS (FOR app.py)
# =========================================================

def insert_prediction(data: dict):
    """
    Wrapper for simple insert without SHAP (used by app.py)
    """
    pred_dict = {
        "timestamp": data.get("timestamp"),
        "employee_id": data.get("employee_id"),
        "run_by": data.get("run_by"),
        "department": data.get("department"),
        "job_role": data.get("job_role"),
        "overtime": data.get("overtime"),

        # minimal required fields
        "probability": data.get("probability"),
        "risk_level": data.get("risk"),
        "prediction": data.get("prediction"),
    }

    # call main function with empty shap
    return save_prediction(pred_dict, [], data.get("threshold", 0.5))


def insert_shap_factors(prediction_id: int, shap_data: list):
    """
    Insert SHAP factors AFTER prediction (used by app.py)
    """
    with get_db() as conn:
        for rank, s in enumerate(shap_data, start=1):
            conn.execute("""
                INSERT INTO shap_factors
                    (prediction_id, rank, feature, shap_value, direction, bar_pct)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                prediction_id,
                rank,
                s.get("feature"),
                s.get("shap_value"),
                s.get("direction", "risk"),
                s.get("bar_pct", 0)
            ))


def get_recent_predictions(limit=5):
    """
    Fetch latest predictions (used in dashboard)
    """
    return get_predictions(limit=limit)