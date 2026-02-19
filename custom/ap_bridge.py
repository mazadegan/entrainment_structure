#!/usr/bin/env python3
"""Build SB-style compatibility DB from custom.db and run python/ap.py metrics.

This reduces drift by reusing the original ap.py implementation.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--custom-db", required=True, help="Path to custom pipeline DB")
    p.add_argument("--compat-db", required=True, help="Path to output SB-compatible DB")
    p.add_argument("--session-id", default=None, help="Optional single-session filter")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory for CSV exports of ap.py results",
    )
    p.add_argument(
        "--nrm-type",
        default="RAW",
        choices=["RAW", "SPEAKER", "GENDER"],
        help="Normalization type passed to ap.load_data",
    )
    return p.parse_args()


def get_session_ids(conn: sqlite3.Connection, session_id: str = None) -> List[str]:
    if session_id:
        rows = conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)).fetchall()
    else:
        rows = conn.execute("SELECT session_id FROM sessions ORDER BY session_id").fetchall()
    return [r[0] for r in rows]


def build_compat_db(custom_db: Path, compat_db: Path, session_id: str = None) -> None:
    root = Path(__file__).resolve().parents[1]
    sql_init = root / "sql" / "init_sb.sql"

    src = sqlite3.connect(str(custom_db))
    dst = sqlite3.connect(str(compat_db))
    try:
        dst.execute("PRAGMA foreign_keys = OFF")
        dst.executescript(sql_init.read_text())
        dst.execute("PRAGMA foreign_keys = ON")
        dst.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_pairs (
                p_or_x TEXT NOT NULL,
                chu_id1 INTEGER NOT NULL,
                chu_id2 INTEGER NOT NULL,
                rid INTEGER,
                FOREIGN KEY (chu_id1) REFERENCES chunks (chu_id),
                FOREIGN KEY (chu_id2) REFERENCES chunks (chu_id)
            )
            """
        )
        dst.execute("CREATE INDEX IF NOT EXISTS chp_chu_fk1 ON chunk_pairs (chu_id1)")
        dst.execute("CREATE INDEX IF NOT EXISTS chp_chu_fk2 ON chunk_pairs (chu_id2)")

        session_ids = get_session_ids(src, session_id)
        if not session_ids:
            raise ValueError("no matching sessions in custom DB")

        next_spk_id = 1
        next_ses_id = 1
        next_tur_id = 1
        next_chu_id = 1
        next_top_id = 1

        for sid in session_ids:
            # Speaker set for this session in chronological order.
            spk_rows = src.execute(
                """
                SELECT speaker_id, MIN(start_s) first_start
                FROM chunks
                WHERE session_id = ?
                GROUP BY speaker_id
                ORDER BY first_start, speaker_id
                """,
                (sid,),
            ).fetchall()
            speakers = [r[0] for r in spk_rows]
            if len(speakers) < 2:
                print(f"SKIP {sid}: fewer than 2 speakers")
                continue
            if len(speakers) > 2:
                print(f"WARN {sid}: {len(speakers)} speakers found; keeping first two only")
            speakers = speakers[:2]

            # Map two speakers to A/B and integer IDs expected by init_sb schema.
            spk_map: Dict[str, int] = {
                speakers[0]: next_spk_id,
                speakers[1]: next_spk_id + 1,
            }
            next_spk_id += 2

            # Use placeholder gender x (unknown).
            dst.execute("INSERT INTO speakers (spk_id, gender) VALUES (?, ?)", (spk_map[speakers[0]], "x"))
            dst.execute("INSERT INTO speakers (spk_id, gender) VALUES (?, ?)", (spk_map[speakers[1]], "x"))

            # Minimal topic/task/session records.
            dst.execute("INSERT INTO topics (top_id, title, details) VALUES (?, ?, ?)", (next_top_id, f"topic_{sid}", ""))
            ses_id_int = next_ses_id
            dst.execute(
                """
                INSERT INTO sessions (ses_id, spk_id_a, spk_id_b, top_id, status, type)
                VALUES (?, ?, ?, ?, 2, 'CONV')
                """,
                (ses_id_int, spk_map[speakers[0]], spk_map[speakers[1]], next_top_id),
            )
            dst.execute(
                "INSERT INTO tasks (tsk_id, ses_id, task_index, a_or_b) VALUES (?, ?, 1, 'A')",
                (ses_id_int, ses_id_int),
            )
            next_top_id += 1
            next_ses_id += 1

            # Pull chunks for two retained speakers only.
            chunk_rows = src.execute(
                """
                SELECT chunk_id, turn_id, speaker_id, start_s, end_s, duration, text,
                       pitch_min, pitch_max, pitch_mean, pitch_std, rate_syl, rate_vcd,
                       intensity_min, intensity_max, intensity_mean, intensity_std,
                       jitter, shimmer, nhr
                FROM chunks
                WHERE session_id = ?
                  AND speaker_id IN (?, ?)
                ORDER BY start_s, end_s, chunk_id
                """,
                (sid, speakers[0], speakers[1]),
            ).fetchall()

            # Rebuild turns in chronological order (global turn_index), one turn per chunk.
            chu_id_map: Dict[str, int] = {}
            turn_idx = 0
            for row in chunk_rows:
                (
                    chunk_id_text,
                    _turn_id_text,
                    speaker_id_text,
                    start_s,
                    end_s,
                    duration,
                    words,
                    pitch_min,
                    pitch_max,
                    pitch_mean,
                    pitch_std,
                    rate_syl,
                    rate_vcd,
                    intensity_min,
                    intensity_max,
                    intensity_mean,
                    intensity_std,
                    jitter,
                    shimmer,
                    nhr,
                ) = row

                turn_idx += 1
                tur_id = next_tur_id
                next_tur_id += 1

                # With task.a_or_b='A': speaker A -> role d, speaker B -> role f
                role = "d" if speaker_id_text == speakers[0] else "f"
                dst.execute(
                    """
                    INSERT INTO turns (tur_id, tsk_id, turn_type, turn_index, turn_index_ses, speaker_role)
                    VALUES (?, ?, NULL, ?, ?, ?)
                    """,
                    (tur_id, ses_id_int, turn_idx, turn_idx, role),
                )

                chu_id = next_chu_id
                next_chu_id += 1
                chu_id_map[chunk_id_text] = chu_id

                dst.execute(
                    """
                    INSERT INTO chunks (
                        chu_id, tur_id, chunk_index, start_time, end_time, duration, words,
                        pitch_min, pitch_max, pitch_mean, pitch_std, rate_syl, rate_vcd,
                        intensity_min, intensity_max, intensity_mean, intensity_std,
                        jitter, shimmer, nhr
                    ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chu_id,
                        tur_id,
                        float(start_s),
                        float(end_s),
                        float(duration),
                        words or "",
                        pitch_min,
                        pitch_max,
                        pitch_mean,
                        pitch_std,
                        rate_syl,
                        rate_vcd,
                        intensity_min,
                        intensity_max,
                        intensity_mean,
                        intensity_std,
                        jitter,
                        shimmer,
                        nhr,
                    ),
                )

            # Map custom chunk_pairs into expected schema (chu_id1/chu_id2/rid).
            pair_rows = src.execute(
                """
                SELECT p_or_x, chunk_id_source, chunk_id_target, rid
                FROM chunk_pairs
                WHERE session_id = ?
                ORDER BY chunk_id_target, p_or_x, rid
                """,
                (sid,),
            ).fetchall()
            for p_or_x, src_id_text, tgt_id_text, rid in pair_rows:
                if src_id_text not in chu_id_map or tgt_id_text not in chu_id_map:
                    continue
                # Original schema stores rid NULL for p.
                rid_val = None if p_or_x == "p" else int(rid)
                dst.execute(
                    "INSERT INTO chunk_pairs (p_or_x, chu_id1, chu_id2, rid) VALUES (?, ?, ?, ?)",
                    (p_or_x, chu_id_map[src_id_text], chu_id_map[tgt_id_text], rid_val),
                )

        dst.commit()
    finally:
        src.close()
        dst.close()


def run_ap_on_compat(compat_db: Path, out_dir: Path = None, nrm_type: str = "RAW") -> None:
    root = Path(__file__).resolve().parents[1]
    py_dir = root / "python"
    if str(py_dir) not in sys.path:
        sys.path.insert(0, str(py_dir))

    import cfg  # type: ignore
    import db  # type: ignore
    import ap  # type: ignore

    # Set runtime paths so db/ap read from compat DB + correct SQL directory.
    cfg.DB_FNAME_SB = str(compat_db)
    cfg.SQL_PATH = str((root / "sql").resolve()) + "/"

    nrm_const = cfg.NRM_RAW if nrm_type == "RAW" else cfg.NRM_SPK if nrm_type == "SPEAKER" else cfg.NRM_GND

    db.connect(cfg.CORPUS_ID_SB)
    try:
        df_bt = ap.load_data(nrm_const)
        df_lsim = ap.lsim(df_bt)
        df_syn = ap.syn(df_bt)
        df_lcon = ap.lcon(df_bt)

        df_spk_pairs = db.pd_read_sql_query(sql_fname=cfg.SQL_SP_FNAME)
        if len(df_spk_pairs) > 0 and set(df_spk_pairs["p_or_x"]) >= {"p", "x"}:
            df_gsim, df_gsim_raw = ap.gsim(df_bt, df_spk_pairs)
        else:
            print("WARN: skipping ap.gsim (speaker_pairs lacks both 'p' and 'x' rows)")
            import pandas as pd  # type: ignore

            df_gsim = pd.DataFrame()
            df_gsim_raw = pd.DataFrame()
        try:
            df_gcon, df_gcon_raw = ap.gcon(df_bt)
        except Exception as exc:
            print(f"WARN: skipping ap.gcon ({exc})")
            import pandas as pd  # type: ignore

            df_gcon = pd.DataFrame()
            df_gcon_raw = pd.DataFrame()
    finally:
        db.close()

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        df_lsim.to_csv(out_dir / "ap_lsim.csv")
        df_syn.to_csv(out_dir / "ap_syn.csv")
        df_lcon.to_csv(out_dir / "ap_lcon.csv")
        df_gsim.to_csv(out_dir / "ap_gsim.csv")
        df_gsim_raw.to_csv(out_dir / "ap_gsim_raw.csv")
        df_gcon.to_csv(out_dir / "ap_gcon.csv")
        df_gcon_raw.to_csv(out_dir / "ap_gcon_raw.csv")

    print("ap.py run complete")
    print(f"  lsim rows={len(df_lsim)}")
    print(f"  syn rows={len(df_syn)}")
    print(f"  lcon rows={len(df_lcon)}")
    print(f"  gsim rows={len(df_gsim)}")
    print(f"  gcon rows={len(df_gcon)}")


def main() -> None:
    args = parse_args()
    custom_db = Path(args.custom_db)
    compat_db = Path(args.compat_db)
    out_dir = Path(args.out_dir) if args.out_dir else None

    if not custom_db.exists():
        raise FileNotFoundError(f"custom DB not found: {custom_db}")

    build_compat_db(custom_db=custom_db, compat_db=compat_db, session_id=args.session_id)
    run_ap_on_compat(compat_db=compat_db, out_dir=out_dir, nrm_type=args.nrm_type)


if __name__ == "__main__":
    main()
