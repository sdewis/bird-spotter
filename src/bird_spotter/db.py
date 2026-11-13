"""SQLite persistence layer for British Bird Spotter sightings, Life List (Bird Dex), and analytics."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any


def compute_quality_score(confidence: float, sharpness: float) -> float:
    """Compute composite photo quality score from species confidence [0..1] and Laplacian sharpness."""
    c = max(0.0, min(1.0, float(confidence)))
    s = max(0.0, float(sharpness))
    return round(c * math.log10(10.0 + s), 4)


class BirdDatabase:
    """Thread-safe SQLite database for bird sightings, species Life List, and photo management."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sightings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    epoch_ts REAL NOT NULL,
                    date_str TEXT NOT NULL,
                    hour_of_day INTEGER NOT NULL,
                    species TEXT NOT NULL,
                    scientific_name TEXT NOT NULL DEFAULT '',
                    yolo_confidence REAL NOT NULL DEFAULT 0.0,
                    species_confidence REAL NOT NULL DEFAULT 0.0,
                    sharpness REAL NOT NULL DEFAULT 0.0,
                    quality_score REAL NOT NULL DEFAULT 0.0,
                    snapshot_filename TEXT,
                    thumb_filename TEXT,
                    zoom_level REAL NOT NULL DEFAULT 1.0,
                    is_manual INTEGER NOT NULL DEFAULT 0,
                    verified_status TEXT NOT NULL DEFAULT 'auto',
                    top_predictions_json TEXT NOT NULL DEFAULT '[]'
                );

                CREATE INDEX IF NOT EXISTS idx_sightings_species ON sightings(species);
                CREATE INDEX IF NOT EXISTS idx_sightings_date ON sightings(date_str);
                CREATE INDEX IF NOT EXISTS idx_sightings_epoch ON sightings(epoch_ts DESC);

                CREATE TABLE IF NOT EXISTS species_dex (
                    species TEXT PRIMARY KEY,
                    scientific_name TEXT NOT NULL DEFAULT '',
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    total_sightings INTEGER NOT NULL DEFAULT 1,
                    best_sighting_id INTEGER,
                    best_snapshot_filename TEXT,
                    best_thumb_filename TEXT,
                    best_confidence REAL NOT NULL DEFAULT 0.0,
                    best_sharpness REAL NOT NULL DEFAULT 0.0,
                    best_quality_score REAL NOT NULL DEFAULT 0.0,
                    last_snapshot_ts REAL NOT NULL DEFAULT 0.0
                );
                """
            )

    def should_save_auto_photo(
        self,
        species: str,
        quality_score: float,
        now_epoch: float | None = None,
        per_species_cooldown_sec: float = 300.0,
    ) -> tuple[bool, str]:
        """Determine whether an automated bird sighting should trigger saving a new photograph.

        Returns (should_save, reason) where reason is one of:
        - 'new_species': Species has never been photographed before
        - 'better_quality': Current frame beats the existing best photo quality score
        - 'cooldown_elapsed': Per-species cooldown has elapsed since last photo
        - 'cooldown_active': Recently photographed and quality does not beat existing best
        """
        now_val = now_epoch if now_epoch is not None else time.time()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT best_snapshot_filename, best_quality_score, last_snapshot_ts
                FROM species_dex
                WHERE species = ?
                """,
                (species,),
            ).fetchone()

            if row is None or not row["best_snapshot_filename"]:
                return True, "new_species"

            best_q = float(row["best_quality_score"] or 0.0)
            last_ts = float(row["last_snapshot_ts"] or 0.0)

            if quality_score > best_q * 1.02:
                return True, "better_quality"

            if (now_val - last_ts) >= per_species_cooldown_sec:
                return True, "cooldown_elapsed"

            return False, "cooldown_active"

    def record_sighting(
        self,
        species: str,
        scientific_name: str = "",
        yolo_confidence: float = 0.0,
        species_confidence: float = 0.0,
        sharpness: float = 0.0,
        quality_score: float | None = None,
        snapshot_filename: str | None = None,
        thumb_filename: str | None = None,
        zoom_level: float = 1.0,
        is_manual: bool = False,
        verified_status: str = "auto",
        top_predictions: list[dict[str, Any]] | None = None,
        now_dt: datetime | None = None,
    ) -> dict[str, Any]:
        """Insert a sighting record and update the species Life List (species_dex)."""
        dt = now_dt or datetime.now()
        ts_iso = dt.strftime("%Y-%m-%d %H:%M:%S")
        date_str = dt.strftime("%Y-%m-%d")
        hour_of_day = dt.hour
        epoch_ts = dt.timestamp()

        eff_conf = species_confidence if species_confidence > 0 else yolo_confidence
        q_score = (
            float(quality_score)
            if quality_score is not None
            else compute_quality_score(eff_conf, sharpness)
        )
        top_json = json.dumps(top_predictions or [])

        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO sightings (
                    timestamp, epoch_ts, date_str, hour_of_day, species, scientific_name,
                    yolo_confidence, species_confidence, sharpness, quality_score,
                    snapshot_filename, thumb_filename, zoom_level, is_manual,
                    verified_status, top_predictions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_iso,
                    epoch_ts,
                    date_str,
                    hour_of_day,
                    species,
                    scientific_name,
                    round(float(yolo_confidence), 4),
                    round(float(species_confidence), 4),
                    round(float(sharpness), 2),
                    round(q_score, 4),
                    snapshot_filename,
                    thumb_filename,
                    round(float(zoom_level), 2),
                    1 if is_manual else 0,
                    verified_status,
                    top_json,
                ),
            )
            sighting_id = int(cur.lastrowid)

            dex_row = conn.execute(
                "SELECT * FROM species_dex WHERE species = ?", (species,)
            ).fetchone()

            if dex_row is None:
                conn.execute(
                    """
                    INSERT INTO species_dex (
                        species, scientific_name, first_seen, last_seen, total_sightings,
                        best_sighting_id, best_snapshot_filename, best_thumb_filename,
                        best_confidence, best_sharpness, best_quality_score, last_snapshot_ts
                    ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        species,
                        scientific_name,
                        ts_iso,
                        ts_iso,
                        sighting_id if snapshot_filename else None,
                        snapshot_filename,
                        thumb_filename,
                        round(float(eff_conf), 4),
                        round(float(sharpness), 2),
                        round(q_score, 4) if snapshot_filename else 0.0,
                        epoch_ts if snapshot_filename else 0.0,
                    ),
                )
            else:
                new_total = int(dex_row["total_sightings"]) + 1
                sci_name = scientific_name or dex_row["scientific_name"]
                best_id = dex_row["best_sighting_id"]
                best_snap = dex_row["best_snapshot_filename"]
                best_thumb = dex_row["best_thumb_filename"]
                best_conf = max(float(dex_row["best_confidence"] or 0.0), float(eff_conf))
                best_sharp = max(float(dex_row["best_sharpness"] or 0.0), float(sharpness))
                best_q = float(dex_row["best_quality_score"] or 0.0)
                last_snap_ts = float(dex_row["last_snapshot_ts"] or 0.0)

                if snapshot_filename:
                    last_snap_ts = epoch_ts
                    if not best_snap or q_score >= best_q:
                        best_id = sighting_id
                        best_snap = snapshot_filename
                        best_thumb = thumb_filename
                        best_q = q_score

                conn.execute(
                    """
                    UPDATE species_dex
                    SET scientific_name = ?,
                        last_seen = ?,
                        total_sightings = ?,
                        best_sighting_id = ?,
                        best_snapshot_filename = ?,
                        best_thumb_filename = ?,
                        best_confidence = ?,
                        best_sharpness = ?,
                        best_quality_score = ?,
                        last_snapshot_ts = ?
                    WHERE species = ?
                    """,
                    (
                        sci_name,
                        ts_iso,
                        new_total,
                        best_id,
                        best_snap,
                        best_thumb,
                        round(best_conf, 4),
                        round(best_sharp, 2),
                        round(best_q, 4),
                        last_snap_ts,
                        species,
                    ),
                )

            return {
                "id": sighting_id,
                "timestamp": ts_iso,
                "species": species,
                "scientific_name": scientific_name,
                "yolo_confidence": round(float(yolo_confidence), 4),
                "species_confidence": round(float(species_confidence), 4),
                "sharpness": round(float(sharpness), 2),
                "quality_score": round(q_score, 4),
                "snapshot_filename": snapshot_filename,
                "thumb_filename": thumb_filename,
                "zoom_level": round(float(zoom_level), 2),
                "is_manual": is_manual,
                "verified_status": verified_status,
            }

    def rebuild_species_dex(self) -> None:
        """Recompute species_dex from sightings table (used after relabel or delete)."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM species_dex;")
            rows = conn.execute(
                "SELECT * FROM sightings ORDER BY epoch_ts ASC, id ASC"
            ).fetchall()

            dex_map: dict[str, dict[str, Any]] = {}
            for r in rows:
                sp = r["species"]
                eff_conf = (
                    float(r["species_confidence"])
                    if float(r["species_confidence"]) > 0
                    else float(r["yolo_confidence"])
                )
                q_score = float(r["quality_score"] or 0.0)
                snap = r["snapshot_filename"]
                thumb = r["thumb_filename"]

                if sp not in dex_map:
                    dex_map[sp] = {
                        "species": sp,
                        "scientific_name": r["scientific_name"] or "",
                        "first_seen": r["timestamp"],
                        "last_seen": r["timestamp"],
                        "total_sightings": 1,
                        "best_sighting_id": r["id"] if snap else None,
                        "best_snapshot_filename": snap,
                        "best_thumb_filename": thumb,
                        "best_confidence": eff_conf,
                        "best_sharpness": float(r["sharpness"] or 0.0),
                        "best_quality_score": q_score if snap else 0.0,
                        "last_snapshot_ts": float(r["epoch_ts"]) if snap else 0.0,
                    }
                else:
                    d = dex_map[sp]
                    d["total_sightings"] += 1
                    d["last_seen"] = r["timestamp"]
                    if r["scientific_name"]:
                        d["scientific_name"] = r["scientific_name"]
                    d["best_confidence"] = max(d["best_confidence"], eff_conf)
                    d["best_sharpness"] = max(d["best_sharpness"], float(r["sharpness"] or 0.0))
                    if snap:
                        d["last_snapshot_ts"] = float(r["epoch_ts"])
                        if not d["best_snapshot_filename"] or q_score >= d["best_quality_score"]:
                            d["best_sighting_id"] = r["id"]
                            d["best_snapshot_filename"] = snap
                            d["best_thumb_filename"] = thumb
                            d["best_quality_score"] = q_score

            for d in dex_map.values():
                conn.execute(
                    """
                    INSERT INTO species_dex (
                        species, scientific_name, first_seen, last_seen, total_sightings,
                        best_sighting_id, best_snapshot_filename, best_thumb_filename,
                        best_confidence, best_sharpness, best_quality_score, last_snapshot_ts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        d["species"],
                        d["scientific_name"],
                        d["first_seen"],
                        d["last_seen"],
                        d["total_sightings"],
                        d["best_sighting_id"],
                        d["best_snapshot_filename"],
                        d["best_thumb_filename"],
                        round(d["best_confidence"], 4),
                        round(d["best_sharpness"], 2),
                        round(d["best_quality_score"], 4),
                        d["last_snapshot_ts"],
                    ),
                )

    def relabel_sighting(
        self, sighting_id: int, new_species: str, scientific_name: str = ""
    ) -> bool:
        """Manually re-label a sighting to a different species and rebuild species_dex."""
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    UPDATE sightings
                    SET species = ?,
                        scientific_name = CASE WHEN ? != '' THEN ? ELSE scientific_name END,
                        verified_status = 'relabeled'
                    WHERE id = ?
                    """,
                    (new_species, scientific_name, scientific_name, int(sighting_id)),
                )
                changed = cur.rowcount > 0
            if changed:
                self.rebuild_species_dex()
            return changed

    def delete_sighting(
        self, sighting_id: int, snapshots_dir: Path | None = None
    ) -> bool:
        """Delete a false-positive sighting, remove its snapshot files, and rebuild species_dex."""
        with self._lock:
            snap_file = None
            thumb_file = None
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT snapshot_filename, thumb_filename FROM sightings WHERE id = ?",
                    (int(sighting_id),),
                ).fetchone()
                if row is None:
                    return False
                snap_file = row["snapshot_filename"]
                thumb_file = row["thumb_filename"]
                conn.execute("DELETE FROM sightings WHERE id = ?", (int(sighting_id),))

            if snapshots_dir is not None:
                for fname in {snap_file, thumb_file}:
                    if fname:
                        fpath = (snapshots_dir / fname).resolve()
                        if fpath.is_file() and snapshots_dir.resolve() in fpath.parents:
                            try:
                                fpath.unlink()
                            except OSError:
                                pass

            self.rebuild_species_dex()
            return True

    def clear_all(self) -> None:
        """Clear all sightings and species_dex records."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM sightings;")
            conn.execute("DELETE FROM species_dex;")

    def get_species_dex(self) -> list[dict[str, Any]]:
        """Return the British Bird Life List (Bird Dex) ordered by most recently seen."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM species_dex
                ORDER BY total_sightings DESC, last_seen DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def get_species_stats_dict(self) -> dict[str, dict[str, Any]]:
        """Return legacy-compatible dictionary keyed by species name."""
        dex = self.get_species_dex()
        result: dict[str, dict[str, Any]] = {}
        for item in dex:
            result[item["species"]] = {
                "count": item["total_sightings"],
                "scientific_name": item["scientific_name"],
                "first_seen": item["first_seen"],
                "last_seen": item["last_seen"],
                "last_confidence": item["best_confidence"],
                "best_snapshot": item["best_snapshot_filename"],
                "best_thumb": item["best_thumb_filename"],
                "best_sharpness": item["best_sharpness"],
                "best_quality_score": item["best_quality_score"],
            }
        return result

    def get_recent_sightings(
        self, limit: int = 24, photos_only: bool = True
    ) -> list[dict[str, Any]]:
        """Return recent sightings with parsed top_predictions."""
        with self._lock, self._connect() as conn:
            where_clause = "WHERE snapshot_filename IS NOT NULL AND snapshot_filename != ''" if photos_only else ""
            rows = conn.execute(
                f"""
                SELECT * FROM sightings
                {where_clause}
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                try:
                    d["top_predictions"] = json.loads(d.pop("top_predictions_json", "[]"))
                except Exception:
                    d["top_predictions"] = []
                d["filename"] = d["snapshot_filename"]
                d["time"] = d["timestamp"].split(" ")[-1] if " " in d["timestamp"] else d["timestamp"]
                out.append(d)
            return out

    def get_hourly_distribution(self, days: int = 30) -> list[int]:
        """Return 24-bin list of sightings per hour of day [00..23] over the past N days."""
        cutoff = (datetime.now() - timedelta(days=days)).timestamp()
        bins = [0] * 24
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT hour_of_day, COUNT(*) AS cnt
                FROM sightings
                WHERE epoch_ts >= ?
                GROUP BY hour_of_day
                """,
                (cutoff,),
            ).fetchall()
            for r in rows:
                h = int(r["hour_of_day"])
                if 0 <= h < 24:
                    bins[h] = int(r["cnt"])
        return bins

    def get_daily_trends(self, days: int = 14) -> list[dict[str, Any]]:
        """Return daily sighting totals and unique species counts over the past N days."""
        today = datetime.now().date()
        date_keys = [
            (today - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(days - 1, -1, -1)
        ]
        day_map = {
            d: {"date": d, "label": d[5:], "total": 0, "unique_species": 0}
            for d in date_keys
        }
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT date_str, COUNT(*) AS total, COUNT(DISTINCT species) AS uniq
                FROM sightings
                WHERE date_str >= ?
                GROUP BY date_str
                """,
                (date_keys[0],),
            ).fetchall()
            for r in rows:
                ds = r["date_str"]
                if ds in day_map:
                    day_map[ds]["total"] = int(r["total"])
                    day_map[ds]["unique_species"] = int(r["uniq"])
        return [day_map[d] for d in date_keys]
