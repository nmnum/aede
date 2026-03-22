"""
Module 5: Decision Trace Logger
================================
Append-only SQLite log. Records every decision alongside the full
model state that generated it. Scientific integrity requirement:
no entry is ever modified after writing.

Schema:
  campaigns     — one row per campaign (config hash, metadata)
  iterations    — one row per iteration
  decisions     — one row per candidate considered per iteration
  observations  — one row per IVO observation received
  insights      — one row per Insight Engine report
"""

from __future__ import annotations
import json
import sqlite3
import time
import numpy as np
from pathlib import Path
from typing import List, Optional, Any, Dict
from contextlib import contextmanager

from .decision_engine import DecisionManifest, DecisionRecord
from .ivo_layer import ObservationRecord


_CREATE_CAMPAIGNS = """
CREATE TABLE IF NOT EXISTS campaigns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    config_hash TEXT    NOT NULL UNIQUE,
    name        TEXT,
    spec_json   TEXT,
    created_at  REAL    NOT NULL,
    status      TEXT    DEFAULT 'running'
);"""

_CREATE_ITERATIONS = """
CREATE TABLE IF NOT EXISTS iterations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_hash       TEXT    NOT NULL,
    iteration           INTEGER NOT NULL,
    timestamp           REAL    NOT NULL,
    pareto_hypervolume  REAL,
    eig_per_cost        REAL,
    budget_remaining    INTEGER,
    should_stop         INTEGER DEFAULT 0,
    stop_reason         TEXT,
    strategy_weights    TEXT,
    surrogate_state     TEXT,
    UNIQUE(campaign_hash, iteration)
);"""

_CREATE_DECISIONS = """
CREATE TABLE IF NOT EXISTS decisions (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_hash               TEXT    NOT NULL,
    iteration                   INTEGER NOT NULL,
    candidate_id                TEXT,
    provenance                  TEXT,
    X                           TEXT,
    decision_score              REAL,
    expected_pareto_improvement REAL,
    uncertainty_discount        REAL,
    feasibility_probability     REAL,
    experiment_cost             REAL,
    rationale                   TEXT,
    rank                        INTEGER,
    selected                    INTEGER DEFAULT 0,
    gittins_index               REAL
);"""

_CREATE_OBSERVATIONS = """
CREATE TABLE IF NOT EXISTS observations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_hash       TEXT    NOT NULL,
    iteration           INTEGER NOT NULL,
    candidate_id        TEXT,
    X                   TEXT,
    objectives          TEXT,
    constraint_valid    INTEGER,
    experiment_cost     REAL,
    prediction_error    TEXT,
    surrogate_prediction TEXT,
    timestamp           REAL    NOT NULL
);"""

_CREATE_INSIGHTS = """
CREATE TABLE IF NOT EXISTS insights (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_hash TEXT NOT NULL,
    created_at    REAL NOT NULL,
    report_json   TEXT
);"""


def _arr(x: Any) -> str:
    """Serialise numpy array or list to compact JSON string."""
    if x is None:
        return "null"
    if isinstance(x, np.ndarray):
        return json.dumps(x.tolist())
    return json.dumps(x)


class DecisionTraceLogger:

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        with self._conn:
            for ddl in [_CREATE_CAMPAIGNS, _CREATE_ITERATIONS, _CREATE_DECISIONS, _CREATE_OBSERVATIONS, _CREATE_INSIGHTS]:
                self._conn.execute(ddl)

    # ── Campaign ──────────────────────────────────────────────────────────────

    def register_campaign(self, config_hash: str, name: str, spec_dict: dict):
        with self._conn:
            try:
                self._conn.execute(
                    "INSERT INTO campaigns (config_hash, name, spec_json, created_at) VALUES (?,?,?,?)",
                    (config_hash, name, json.dumps(spec_dict, default=str), time.time()),
                )
            except sqlite3.IntegrityError:
                pass  # Campaign already registered

    def mark_campaign_done(self, config_hash: str, status: str = "completed"):
        with self._conn:
            self._conn.execute(
                "UPDATE campaigns SET status=? WHERE config_hash=?",
                (status, config_hash),
            )

    # ── Iteration ─────────────────────────────────────────────────────────────

    def log_iteration(
        self,
        campaign_hash: str,
        manifest: DecisionManifest,
        surrogate_state: Optional[dict] = None,
    ):
        weights_json = json.dumps(manifest.strategy_weights)
        surrogate_json = json.dumps(surrogate_state or {})

        with self._conn:
            # Upsert iteration record
            self._conn.execute(
                """INSERT OR REPLACE INTO iterations
                   (campaign_hash, iteration, timestamp, pareto_hypervolume,
                    eig_per_cost, budget_remaining, should_stop, stop_reason,
                    strategy_weights, surrogate_state)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    campaign_hash,
                    manifest.iteration,
                    time.time(),
                    manifest.pareto_hypervolume,
                    manifest.eig_per_cost,
                    manifest.budget_remaining,
                    int(manifest.should_stop),
                    manifest.stop_reason,
                    weights_json,
                    surrogate_json,
                ),
            )
            selected_ids = {r.candidate_id for r in manifest.selected}

            # Log all scored candidates
            for rec in manifest.all_scored:
                self._conn.execute(
                    """INSERT INTO decisions
                       (campaign_hash, iteration, candidate_id, provenance, X,
                        decision_score, expected_pareto_improvement, uncertainty_discount,
                        feasibility_probability, experiment_cost, rationale, rank,
                        selected, gittins_index)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        campaign_hash,
                        manifest.iteration,
                        rec.candidate_id,
                        rec.provenance,
                        _arr(rec.X),
                        rec.decision_score,
                        rec.expected_pareto_improvement,
                        rec.uncertainty_discount,
                        rec.feasibility_probability,
                        rec.experiment_cost,
                        rec.rationale,
                        rec.rank,
                        int(rec.candidate_id in selected_ids),
                        rec.gittins_index,
                    ),
                )

    def log_observations(
        self,
        campaign_hash: str,
        iteration: int,
        observations: List[ObservationRecord],
    ):
        with self._conn:
            for obs in observations:
                self._conn.execute(
                    """INSERT INTO observations
                       (campaign_hash, iteration, candidate_id, X, objectives,
                        constraint_valid, experiment_cost, prediction_error,
                        surrogate_prediction, timestamp)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        campaign_hash,
                        iteration,
                        obs.candidate_id,
                        _arr(obs.X),
                        _arr(obs.objectives),
                        int(obs.constraint_valid),
                        obs.experiment_cost,
                        _arr(obs.prediction_error),
                        _arr(obs.surrogate_prediction),
                        time.time(),
                    ),
                )

    def log_insight(self, campaign_hash: str, report: dict):
        with self._conn:
            self._conn.execute(
                "INSERT INTO insights (campaign_hash, created_at, report_json) VALUES (?,?,?)",
                (campaign_hash, time.time(), json.dumps(report, default=str)),
            )

    # ── Queries for dashboard and Insight Engine ──────────────────────────────

    def get_hv_history(self, campaign_hash: str) -> List[dict]:
        rows = self._conn.execute(
            "SELECT iteration, pareto_hypervolume, eig_per_cost, budget_remaining "
            "FROM iterations WHERE campaign_hash=? ORDER BY iteration",
            (campaign_hash,),
        ).fetchall()
        return [
            {"iteration": r[0], "hv": r[1], "eig_per_cost": r[2], "budget_remaining": r[3]}
            for r in rows
        ]

    def get_strategy_weight_history(self, campaign_hash: str) -> List[dict]:
        rows = self._conn.execute(
            "SELECT iteration, strategy_weights FROM iterations WHERE campaign_hash=? ORDER BY iteration",
            (campaign_hash,),
        ).fetchall()
        result = []
        for iteration, w_json in rows:
            weights = json.loads(w_json) if w_json else {}
            result.append({"iteration": iteration, **weights})
        return result

    def get_observations(self, campaign_hash: str) -> List[dict]:
        rows = self._conn.execute(
            "SELECT candidate_id, objectives, constraint_valid, experiment_cost, iteration "
            "FROM observations WHERE campaign_hash=? ORDER BY timestamp",
            (campaign_hash,),
        ).fetchall()
        obs = []
        for r in rows:
            objectives = json.loads(r[1]) if r[1] and r[1] != "null" else []
            obs.append({
                "candidate_id": r[0],
                "objectives": objectives,
                "constraint_valid": bool(r[2]),
                "cost": r[3],
                "iteration": r[4],
            })
        return obs

    def get_recent_decisions(self, campaign_hash: str, n: int = 20) -> List[dict]:
        rows = self._conn.execute(
            """SELECT d.candidate_id, d.provenance, d.decision_score,
                      d.expected_pareto_improvement, d.feasibility_probability,
                      d.experiment_cost, d.rationale, d.rank, d.selected, d.iteration
               FROM decisions d
               WHERE d.campaign_hash=? AND d.selected=1
               ORDER BY d.iteration DESC, d.rank ASC
               LIMIT ?""",
            (campaign_hash, n),
        ).fetchall()
        cols = ["candidate_id","provenance","decision_score","expected_pareto_improvement",
                "feasibility_probability","experiment_cost","rationale","rank","selected","iteration"]
        return [dict(zip(cols, r)) for r in rows]

    def get_all_campaigns(self) -> List[dict]:
        rows = self._conn.execute(
            "SELECT config_hash, name, created_at, status FROM campaigns ORDER BY created_at DESC"
        ).fetchall()
        return [{"hash": r[0], "name": r[1], "created_at": r[2], "status": r[3]} for r in rows]

    def get_latest_insight(self, campaign_hash: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT report_json FROM insights WHERE campaign_hash=? ORDER BY created_at DESC LIMIT 1",
            (campaign_hash,),
        ).fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return None

    def close(self):
        self._conn.close()

    @classmethod
    def from_file(cls, path: str) -> "DecisionTraceLogger":
        """Open an existing trace database."""
        return cls(db_path=path)
