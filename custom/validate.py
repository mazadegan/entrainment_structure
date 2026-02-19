#!/usr/bin/env python3
"""Validate custom pipeline outputs for one or more sessions."""

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

DEFAULT_FEATURES = ["intensity_mean", "pitch_mean", "rate_syl"]
MEASURE_TABLES = {
    "lsim": "lsim_summary",
    "syn": "syn_summary",
    "lcon": "lcon_summary",
    "gsim": "gsim_summary",
    "gcon": "gcon_summary",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, help="Path to SQLite DB")
    p.add_argument("--session-id", default=None, help="Restrict validation to one session")
    p.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated features expected in extraction/measure tables",
    )
    p.add_argument(
        "--min-feature-fill",
        type=float,
        default=1.0,
        help="Minimum fraction of chunks with non-null feature values",
    )
    p.add_argument(
        "--require-measures",
        default="lsim,syn,lcon,gsim,gcon",
        help="Comma-separated measure ids to require",
    )
    return p.parse_args()


def parse_csv(raw: str) -> List[str]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    return vals


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def get_session_ids(conn: sqlite3.Connection, session_id: str = None) -> List[str]:
    if session_id:
        rows = conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)).fetchall()
    else:
        rows = conn.execute("SELECT session_id FROM sessions ORDER BY session_id").fetchall()
    return [r[0] for r in rows]


def check_core_counts(conn: sqlite3.Connection, sid: str) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ok = True
    counts: Dict[str, int] = {}
    for table in ["sessions", "turns", "chunks", "word_alignments"]:
        if not table_exists(conn, table):
            msgs.append(f"FAIL: missing table '{table}'")
            return False, msgs

    counts["sessions"] = conn.execute(
        "SELECT count(*) FROM sessions WHERE session_id = ?", (sid,)
    ).fetchone()[0]
    counts["turns"] = conn.execute(
        "SELECT count(*) FROM turns WHERE session_id = ?", (sid,)
    ).fetchone()[0]
    counts["chunks"] = conn.execute(
        "SELECT count(*) FROM chunks WHERE session_id = ?", (sid,)
    ).fetchone()[0]
    counts["word_alignments"] = conn.execute(
        "SELECT count(*) FROM word_alignments WHERE session_id = ?", (sid,)
    ).fetchone()[0]

    msgs.append(
        "INFO: core counts "
        f"sessions={counts['sessions']} turns={counts['turns']} "
        f"chunks={counts['chunks']} word_alignments={counts['word_alignments']}"
    )

    if counts["sessions"] != 1:
        ok = False
        msgs.append("FAIL: expected exactly one session row")
    if counts["turns"] == 0:
        ok = False
        msgs.append("FAIL: no turns found")
    if counts["chunks"] == 0:
        ok = False
        msgs.append("FAIL: no chunks found")

    return ok, msgs


def check_feature_fill(
    conn: sqlite3.Connection, sid: str, features: Sequence[str], min_fill: float
) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ok = True
    total = conn.execute(
        "SELECT count(*) FROM chunks WHERE session_id = ?", (sid,)
    ).fetchone()[0]
    if total == 0:
        return False, ["FAIL: no chunks available for feature validation"]

    for feat in features:
        if not table_exists(conn, "chunks"):
            return False, ["FAIL: missing table 'chunks'"]
        filled = conn.execute(
            f"SELECT count(*) FROM chunks WHERE session_id = ? AND {feat} IS NOT NULL",
            (sid,),
        ).fetchone()[0]
        ratio = filled / total
        msgs.append(f"INFO: feature fill {feat}={filled}/{total} ({ratio:.3f})")
        if ratio < min_fill:
            ok = False
            msgs.append(f"FAIL: feature fill below threshold for {feat} (< {min_fill:.3f})")
    return ok, msgs


def check_pairs(conn: sqlite3.Connection, sid: str) -> Tuple[bool, List[str]]:
    if not table_exists(conn, "chunk_pairs"):
        return False, ["FAIL: missing table 'chunk_pairs'"]

    msgs: List[str] = []
    ok = True
    counts = dict(
        conn.execute(
            "SELECT p_or_x, count(*) FROM chunk_pairs WHERE session_id = ? GROUP BY p_or_x",
            (sid,),
        ).fetchall()
    )
    p_cnt = int(counts.get("p", 0))
    x_cnt = int(counts.get("x", 0))
    msgs.append(f"INFO: pair counts p={p_cnt} x={x_cnt}")
    if p_cnt == 0:
        ok = False
        msgs.append("FAIL: no adjacent partner pairs (p)")
    if x_cnt == 0:
        ok = False
        msgs.append("FAIL: no non-adjacent baseline pairs (x)")

    paired_targets = conn.execute(
        """
        SELECT count(*)
        FROM (
          SELECT chunk_id_target
          FROM chunk_pairs
          WHERE session_id = ?
          GROUP BY chunk_id_target
          HAVING sum(case when p_or_x='p' then 1 else 0 end) > 0
             AND sum(case when p_or_x='x' then 1 else 0 end) > 0
        )
        """,
        (sid,),
    ).fetchone()[0]
    msgs.append(f"INFO: targets with both p and x = {paired_targets}")
    if paired_targets == 0:
        ok = False
        msgs.append("FAIL: no targets have both p and x pairs")

    return ok, msgs


def check_measure_tables(
    conn: sqlite3.Connection, sid: str, features: Sequence[str], require_measures: Sequence[str]
) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ok = True

    for mea in require_measures:
        if mea not in MEASURE_TABLES:
            ok = False
            msgs.append(f"FAIL: unknown measure id '{mea}'")
            continue

        table = MEASURE_TABLES[mea]
        if not table_exists(conn, table):
            ok = False
            msgs.append(f"FAIL: missing table '{table}' for measure '{mea}'")
            continue

        if mea == "lsim":
            # lsim has both pairs and summary; require summary rows by feature
            rows = conn.execute(
                "SELECT feature, count(*) FROM lsim_summary WHERE session_id = ? GROUP BY feature",
                (sid,),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT feature, count(*) FROM {table} WHERE session_id = ? GROUP BY feature",
                (sid,),
            ).fetchall()

        by_feat = {f: c for f, c in rows}
        missing = [f for f in features if by_feat.get(f, 0) == 0]
        if missing:
            ok = False
            msgs.append(f"FAIL: {mea} missing features: {missing}")
        else:
            msgs.append(f"INFO: {mea} coverage OK for features={list(features)}")

    return ok, msgs


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"db not found: {db_path}")

    features = parse_csv(args.features)
    require_measures = parse_csv(args.require_measures)
    if not features:
        raise ValueError("no features provided")
    if args.min_feature_fill < 0 or args.min_feature_fill > 1:
        raise ValueError("--min-feature-fill must be in [0,1]")

    conn = sqlite3.connect(str(db_path))
    try:
        session_ids = get_session_ids(conn, args.session_id)
        if not session_ids:
            raise ValueError("no matching sessions found")

        any_fail = False
        for sid in session_ids:
            print(f"SESSION {sid}")
            checks = [
                check_core_counts(conn, sid),
                check_feature_fill(conn, sid, features, args.min_feature_fill),
                check_pairs(conn, sid),
                check_measure_tables(conn, sid, features, require_measures),
            ]
            for ok, msgs in checks:
                for msg in msgs:
                    print(f"  {msg}")
                if not ok:
                    any_fail = True

        if any_fail:
            print("VALIDATION: FAIL")
            raise SystemExit(1)
        print("VALIDATION: PASS")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
