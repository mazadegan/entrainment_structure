#!/usr/bin/env python3
"""Compute entrainment metrics from chunk_pairs (lsim, syn, lcon, gsim, gcon modes)."""

import argparse
import math
import sqlite3
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from scipy import stats  # type: ignore
except Exception:
    stats = None

DEFAULT_FEATURES = ["intensity_mean", "pitch_mean", "rate_syl"]
ALL_FEATURES = [
    "intensity_min",
    "intensity_max",
    "intensity_mean",
    "intensity_std",
    "pitch_min",
    "pitch_max",
    "pitch_mean",
    "pitch_std",
    "jitter",
    "shimmer",
    "nhr",
    "rate_syl",
    "rate_vcd",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, help="Path to SQLite DB")
    p.add_argument("--session-id", default=None, help="Restrict computation to one session")
    p.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated feature columns from chunks",
    )
    p.add_argument(
        "--mode",
        choices=["lsim", "syn", "lcon", "gsim", "gcon"],
        default="lsim",
        help="Metric to compute",
    )
    p.add_argument(
        "--replace-session",
        action="store_true",
        help="Delete existing result rows for targeted session(s) before insert",
    )
    return p.parse_args()


def parse_feature_list(raw: str) -> List[str]:
    feats = [f.strip() for f in raw.split(",") if f.strip()]
    invalid = [f for f in feats if f not in ALL_FEATURES]
    if invalid:
        raise ValueError(f"unsupported feature(s): {invalid}")
    if not feats:
        raise ValueError("at least one feature is required")
    return feats


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lsim_pairs (
            session_id TEXT NOT NULL,
            feature TEXT NOT NULL,
            chunk_id_target TEXT NOT NULL,
            sim_p REAL NOT NULL,
            sim_x_mean REAL NOT NULL,
            lsim REAL,
            n_x INTEGER NOT NULL,
            PRIMARY KEY (session_id, feature, chunk_id_target),
            FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
            FOREIGN KEY (chunk_id_target) REFERENCES chunks (chunk_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lsim_summary (
            session_id TEXT NOT NULL,
            feature TEXT NOT NULL,
            n_targets INTEGER NOT NULL,
            mean_sim_p REAL,
            mean_sim_x REAL,
            mean_lsim REAL,
            entrained_count INTEGER NOT NULL,
            t_stat REAL,
            p_value REAL,
            PRIMARY KEY (session_id, feature),
            FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS syn_summary (
            session_id TEXT NOT NULL,
            feature TEXT NOT NULL,
            n_pairs INTEGER NOT NULL,
            r_value REAL,
            p_value REAL,
            dof INTEGER,
            PRIMARY KEY (session_id, feature),
            FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcon_summary (
            session_id TEXT NOT NULL,
            feature TEXT NOT NULL,
            n_points INTEGER NOT NULL,
            r_value REAL,
            p_value REAL,
            dof INTEGER,
            PRIMARY KEY (session_id, feature),
            FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gsim_summary (
            session_id TEXT NOT NULL,
            feature TEXT NOT NULL,
            n_targets INTEGER NOT NULL,
            mean_sim_p REAL,
            mean_sim_x REAL,
            mean_sim_nrm REAL,
            t_stat REAL,
            p_value REAL,
            PRIMARY KEY (session_id, feature),
            FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gcon_summary (
            session_id TEXT NOT NULL,
            feature TEXT NOT NULL,
            n_pairs INTEGER NOT NULL,
            mean_dist_first REAL,
            mean_dist_second REAL,
            mean_con REAL,
            t_stat REAL,
            p_value REAL,
            dof INTEGER,
            PRIMARY KEY (session_id, feature),
            FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
        )
        """
    )


def get_session_ids(conn: sqlite3.Connection, session_id: Optional[str]) -> List[str]:
    if session_id:
        rows = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT session_id FROM sessions ORDER BY session_id").fetchall()
    return [r[0] for r in rows]


def fetch_pair_sims(
    conn: sqlite3.Connection, session_id: str, feature: str
) -> Dict[str, Dict[str, List[float]]]:
    """Return per-target partner and baseline similarity values for one feature."""
    sql = f"""
        SELECT cp.chunk_id_target,
               cp.p_or_x,
               ABS(src.{feature} - tgt.{feature}) AS dist
        FROM chunk_pairs cp
        JOIN chunks src ON cp.chunk_id_source = src.chunk_id
        JOIN chunks tgt ON cp.chunk_id_target = tgt.chunk_id
        WHERE cp.session_id = ?
    """
    rows = conn.execute(sql, (session_id,)).fetchall()

    by_target: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"p": [], "x": []})
    for chunk_id_target, p_or_x, dist in rows:
        if dist is None:
            continue
        sim = -float(dist)
        by_target[chunk_id_target][p_or_x].append(sim)
    return by_target


def summarize_lsim_pairs(
    session_id: str, feature: str, by_target: Dict[str, Dict[str, List[float]]]
) -> Tuple[List[Tuple], Optional[Tuple]]:
    pair_rows: List[Tuple] = []

    sim_ps: List[float] = []
    sim_xs: List[float] = []
    lsims: List[float] = []

    for chunk_id_target, vals in by_target.items():
        if len(vals["p"]) == 0 or len(vals["x"]) == 0:
            continue

        sim_p = statistics.mean(vals["p"])
        sim_x_mean = statistics.mean(vals["x"])
        lsim = None
        if sim_x_mean != 0.0:
            lsim = -sim_p / sim_x_mean

        pair_rows.append(
            (
                session_id,
                feature,
                chunk_id_target,
                sim_p,
                sim_x_mean,
                lsim,
                len(vals["x"]),
            )
        )

        sim_ps.append(sim_p)
        sim_xs.append(sim_x_mean)
        if lsim is not None and not math.isnan(lsim):
            lsims.append(lsim)

    if not pair_rows:
        return pair_rows, None

    t_stat = None
    p_value = None
    if stats is not None and len(sim_ps) > 1:
        t_res = stats.ttest_rel(sim_ps, sim_xs)
        t_stat = float(t_res.statistic)
        p_value = float(t_res.pvalue)

    mean_sim_p = statistics.mean(sim_ps) if sim_ps else None
    mean_sim_x = statistics.mean(sim_xs) if sim_xs else None
    mean_lsim = statistics.mean(lsims) if lsims else None
    entrained_count = sum(1 for x in lsims if x > -1.0)

    summary_row = (
        session_id,
        feature,
        len(pair_rows),
        mean_sim_p,
        mean_sim_x,
        mean_lsim,
        entrained_count,
        t_stat,
        p_value,
    )
    return pair_rows, summary_row


def _pearsonr_basic(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mx = statistics.mean(x)
    my = statistics.mean(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    if sx == 0.0 or sy == 0.0:
        return None
    cov = sum((vx - mx) * (vy - my) for vx, vy in zip(x, y))
    return cov / (sx * sy)


def summarize_syn(
    conn: sqlite3.Connection, session_id: str, feature: str
) -> Optional[Tuple]:
    sql = f"""
        SELECT src.{feature} AS source_val,
               tgt.{feature} AS target_val
        FROM chunk_pairs cp
        JOIN chunks src ON cp.chunk_id_source = src.chunk_id
        JOIN chunks tgt ON cp.chunk_id_target = tgt.chunk_id
        WHERE cp.session_id = ?
          AND cp.p_or_x = 'p'
          AND src.{feature} IS NOT NULL
          AND tgt.{feature} IS NOT NULL
    """
    rows = conn.execute(sql, (session_id,)).fetchall()
    if not rows:
        return None

    x = [float(r[0]) for r in rows]
    y = [float(r[1]) for r in rows]
    n = len(x)
    r_value = _pearsonr_basic(x, y)

    p_value = None
    if stats is not None and n >= 3:
        try:
            pr = stats.pearsonr(x, y)
            r_value = float(pr.statistic)
            p_value = float(pr.pvalue)
        except Exception:
            pass

    dof = n - 2 if n >= 2 else None
    return (session_id, feature, n, r_value, p_value, dof)


def summarize_lcon(
    conn: sqlite3.Connection, session_id: str, feature: str
) -> Optional[Tuple]:
    """Correlation between local similarity and target chunk start time."""
    sql = f"""
        SELECT tgt.start_s AS target_start_s,
               ABS(src.{feature} - tgt.{feature}) AS dist
        FROM chunk_pairs cp
        JOIN chunks src ON cp.chunk_id_source = src.chunk_id
        JOIN chunks tgt ON cp.chunk_id_target = tgt.chunk_id
        WHERE cp.session_id = ?
          AND cp.p_or_x = 'p'
          AND src.{feature} IS NOT NULL
          AND tgt.{feature} IS NOT NULL
    """
    rows = conn.execute(sql, (session_id,)).fetchall()
    if not rows:
        return None

    times: List[float] = []
    sims: List[float] = []
    for start_s, dist in rows:
        if start_s is None or dist is None:
            continue
        times.append(float(start_s))
        sims.append(-float(dist))

    n = len(times)
    if n == 0:
        return None

    r_value = _pearsonr_basic(sims, times)
    p_value = None
    if stats is not None and n >= 3:
        try:
            pr = stats.pearsonr(sims, times)
            r_value = float(pr.statistic)
            p_value = float(pr.pvalue)
        except Exception:
            pass

    dof = n - 2 if n >= 2 else None
    return (session_id, feature, n, r_value, p_value, dof)


def run_lsim(conn: sqlite3.Connection, session_ids: List[str], features: List[str], replace_session: bool) -> None:
    all_pair_rows: List[Tuple] = []
    all_summary_rows: List[Tuple] = []

    for sid in session_ids:
        if replace_session:
            conn.execute("DELETE FROM lsim_pairs WHERE session_id = ?", (sid,))
            conn.execute("DELETE FROM lsim_summary WHERE session_id = ?", (sid,))

        for feature in features:
            by_target = fetch_pair_sims(conn, sid, feature)
            pair_rows, summary_row = summarize_lsim_pairs(sid, feature, by_target)
            all_pair_rows.extend(pair_rows)
            if summary_row is not None:
                all_summary_rows.append(summary_row)

    if all_pair_rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO lsim_pairs
            (session_id, feature, chunk_id_target, sim_p, sim_x_mean, lsim, n_x)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            all_pair_rows,
        )

    if all_summary_rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO lsim_summary
            (session_id, feature, n_targets, mean_sim_p, mean_sim_x, mean_lsim,
             entrained_count, t_stat, p_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            all_summary_rows,
        )

    conn.commit()

    print(
        f"lsim complete: sessions={len(session_ids)} features={len(features)} "
        f"pair_rows={len(all_pair_rows)} summary_rows={len(all_summary_rows)}"
    )
    for row in all_summary_rows:
        sid, feat, n, mean_p, mean_x, mean_l, ent_cnt, t_stat, p_val = row
        print(
            f"  {sid} {feat}: n={n} mean_sim_p={mean_p:.4f} "
            f"mean_sim_x={mean_x:.4f} mean_lsim={mean_l:.4f} "
            f"entrained={ent_cnt}/{n} t={t_stat} p={p_val}"
        )


def run_syn(conn: sqlite3.Connection, session_ids: List[str], features: List[str], replace_session: bool) -> None:
    summary_rows: List[Tuple] = []

    for sid in session_ids:
        if replace_session:
            conn.execute("DELETE FROM syn_summary WHERE session_id = ?", (sid,))

        for feature in features:
            row = summarize_syn(conn, sid, feature)
            if row is not None:
                summary_rows.append(row)

    if summary_rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO syn_summary
            (session_id, feature, n_pairs, r_value, p_value, dof)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            summary_rows,
        )
    conn.commit()

    print(
        f"syn complete: sessions={len(session_ids)} features={len(features)} "
        f"summary_rows={len(summary_rows)}"
    )
    for sid, feat, n, r, p, dof in summary_rows:
        print(f"  {sid} {feat}: n={n} r={r} p={p} dof={dof}")


def run_lcon(conn: sqlite3.Connection, session_ids: List[str], features: List[str], replace_session: bool) -> None:
    summary_rows: List[Tuple] = []

    for sid in session_ids:
        if replace_session:
            conn.execute("DELETE FROM lcon_summary WHERE session_id = ?", (sid,))

        for feature in features:
            row = summarize_lcon(conn, sid, feature)
            if row is not None:
                summary_rows.append(row)

    if summary_rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO lcon_summary
            (session_id, feature, n_points, r_value, p_value, dof)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            summary_rows,
        )
    conn.commit()

    print(
        f"lcon complete: sessions={len(session_ids)} features={len(features)} "
        f"summary_rows={len(summary_rows)}"
    )
    for sid, feat, n, r, p, dof in summary_rows:
        print(f"  {sid} {feat}: n={n} r={r} p={p} dof={dof}")


def run_gsim(conn: sqlite3.Connection, session_ids: List[str], features: List[str], replace_session: bool) -> None:
    summary_rows: List[Tuple] = []

    for sid in session_ids:
        if replace_session:
            conn.execute("DELETE FROM gsim_summary WHERE session_id = ?", (sid,))

        for feature in features:
            by_target = fetch_pair_sims(conn, sid, feature)

            sim_ps: List[float] = []
            sim_xs: List[float] = []
            for vals in by_target.values():
                if len(vals["p"]) == 0 or len(vals["x"]) == 0:
                    continue
                sim_ps.append(statistics.mean(vals["p"]))
                sim_xs.append(statistics.mean(vals["x"]))

            if not sim_ps:
                continue

            mean_sim_p = statistics.mean(sim_ps)
            mean_sim_x = statistics.mean(sim_xs)
            mean_sim_nrm = None
            if mean_sim_x != 0.0:
                mean_sim_nrm = -mean_sim_p / mean_sim_x

            t_stat = None
            p_value = None
            if stats is not None and len(sim_ps) > 1:
                t_res = stats.ttest_rel(sim_ps, sim_xs)
                t_stat = float(t_res.statistic)
                p_value = float(t_res.pvalue)

            summary_rows.append(
                (
                    sid,
                    feature,
                    len(sim_ps),
                    mean_sim_p,
                    mean_sim_x,
                    mean_sim_nrm,
                    t_stat,
                    p_value,
                )
            )

    if summary_rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO gsim_summary
            (session_id, feature, n_targets, mean_sim_p, mean_sim_x, mean_sim_nrm, t_stat, p_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            summary_rows,
        )
    conn.commit()

    print(
        f"gsim complete: sessions={len(session_ids)} features={len(features)} "
        f"summary_rows={len(summary_rows)}"
    )
    for sid, feat, n, msp, msx, msn, t, p in summary_rows:
        print(
            f"  {sid} {feat}: n={n} mean_sim_p={msp:.4f} "
            f"mean_sim_x={msx:.4f} mean_sim_nrm={msn:.4f} t={t} p={p}"
        )


def _speaker_means_for_half(
    conn: sqlite3.Connection, session_id: str, feature: str, start_lo: float, start_hi: float
) -> Dict[str, float]:
    sql = f"""
        SELECT speaker_id, AVG({feature}) AS mean_feature
        FROM chunks
        WHERE session_id = ?
          AND start_s >= ?
          AND start_s < ?
          AND {feature} IS NOT NULL
        GROUP BY speaker_id
    """
    rows = conn.execute(sql, (session_id, start_lo, start_hi)).fetchall()
    return {spk: float(mean_val) for spk, mean_val in rows if mean_val is not None}


def summarize_gcon(conn: sqlite3.Connection, session_id: str, feature: str) -> Optional[Tuple]:
    # Use target chunk start-time midpoint as split between halves.
    bounds = conn.execute(
        """
        SELECT MIN(start_s), MAX(start_s)
        FROM chunks
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if bounds is None or bounds[0] is None or bounds[1] is None:
        return None
    lo, hi = float(bounds[0]), float(bounds[1])
    if hi <= lo:
        return None
    mid = (lo + hi) / 2.0

    means_first = _speaker_means_for_half(conn, session_id, feature, lo, mid + 1e-12)
    means_second = _speaker_means_for_half(conn, session_id, feature, mid, hi + 1e-12)
    common_spks = sorted(set(means_first.keys()) & set(means_second.keys()))
    if len(common_spks) < 2:
        return None

    d1: List[float] = []
    d2: List[float] = []
    for spk_a, spk_b in combinations(common_spks, 2):
        d1.append(abs(means_first[spk_a] - means_first[spk_b]))
        d2.append(abs(means_second[spk_a] - means_second[spk_b]))

    if not d1:
        return None

    mean_dist_first = statistics.mean(d1)
    mean_dist_second = statistics.mean(d2)
    con_vals = [a - b for a, b in zip(d1, d2)]
    mean_con = statistics.mean(con_vals)

    t_stat = None
    p_value = None
    if stats is not None and len(d1) > 1:
        t_res = stats.ttest_rel(d1, d2)
        t_stat = float(t_res.statistic)
        p_value = float(t_res.pvalue)

    dof = len(d1) - 1 if len(d1) >= 1 else None
    return (
        session_id,
        feature,
        len(d1),
        mean_dist_first,
        mean_dist_second,
        mean_con,
        t_stat,
        p_value,
        dof,
    )


def run_gcon(conn: sqlite3.Connection, session_ids: List[str], features: List[str], replace_session: bool) -> None:
    summary_rows: List[Tuple] = []

    for sid in session_ids:
        if replace_session:
            conn.execute("DELETE FROM gcon_summary WHERE session_id = ?", (sid,))

        for feature in features:
            row = summarize_gcon(conn, sid, feature)
            if row is not None:
                summary_rows.append(row)

    if summary_rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO gcon_summary
            (session_id, feature, n_pairs, mean_dist_first, mean_dist_second,
             mean_con, t_stat, p_value, dof)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            summary_rows,
        )
    conn.commit()

    print(
        f"gcon complete: sessions={len(session_ids)} features={len(features)} "
        f"summary_rows={len(summary_rows)}"
    )
    for sid, feat, n, d1, d2, con, t, p, dof in summary_rows:
        print(
            f"  {sid} {feat}: n={n} mean_dist_first={d1:.4f} "
            f"mean_dist_second={d2:.4f} mean_con={con:.4f} t={t} p={p} dof={dof}"
        )


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"db not found: {db_path}")

    features = parse_feature_list(args.features)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        ensure_schema(conn)

        session_ids = get_session_ids(conn, args.session_id)
        if not session_ids:
            raise ValueError("no matching sessions found")

        if args.mode == "lsim":
            run_lsim(conn, session_ids, features, args.replace_session)
        elif args.mode == "syn":
            run_syn(conn, session_ids, features, args.replace_session)
        elif args.mode == "lcon":
            run_lcon(conn, session_ids, features, args.replace_session)
        elif args.mode == "gsim":
            run_gsim(conn, session_ids, features, args.replace_session)
        else:
            run_gcon(conn, session_ids, features, args.replace_session)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
