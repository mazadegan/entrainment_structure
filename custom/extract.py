#!/usr/bin/env python3
"""Extract acoustic-prosodic features for ingested chunks and update SQLite."""

import argparse
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from syllables import count_syllables


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, help="Path to SQLite DB")
    p.add_argument("--session-id", default=None, help="Restrict extraction to one session")
    p.add_argument(
        "--praat-script",
        default=str(Path(__file__).resolve().parents[1] / "praat" / "extract_features.praat"),
        help="Path to praat extraction script",
    )
    p.add_argument(
        "--tmp-dir",
        default=tempfile.gettempdir(),
        help="Directory for temporary wav/txt files",
    )
    p.add_argument(
        "--min-duration",
        type=float,
        default=0.04,
        help="Minimum chunk duration (seconds) required for extraction",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute features even if chunk already has pitch_mean",
    )
    return p.parse_args()


def ensure_tools() -> None:
    missing = [tool for tool in ("sox", "praat") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"missing required tool(s): {', '.join(missing)}")


def read_praat_output(path: Path) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        key, val = line.split(",", 1)
        try:
            feats[key] = float(val)
        except ValueError:
            feats[key] = None  # type: ignore[assignment]
    return feats


def iter_targets(conn: sqlite3.Connection, session_id: str = None) -> Iterable[Tuple[str, str, str, float, float, str, float]]:
    sql = (
        "SELECT c.chunk_id, c.session_id, s.audio_path, c.start_s, c.end_s, c.text, c.pitch_mean "
        "FROM chunks c JOIN sessions s ON c.session_id = s.session_id "
    )
    params: List[str] = []
    if session_id:
        sql += "WHERE c.session_id = ? "
        params.append(session_id)
    sql += "ORDER BY c.session_id, c.start_s, c.chunk_id"
    cur = conn.execute(sql, params)
    for row in cur.fetchall():
        yield row


def update_chunk_features(conn: sqlite3.Connection, chunk_id: str, features: Dict[str, float]) -> None:
    conn.execute(
        """
        UPDATE chunks
        SET pitch_min = ?,
            pitch_max = ?,
            pitch_mean = ?,
            pitch_std = ?,
            rate_syl = ?,
            rate_vcd = ?,
            intensity_min = ?,
            intensity_max = ?,
            intensity_mean = ?,
            intensity_std = ?,
            jitter = ?,
            shimmer = ?,
            nhr = ?
        WHERE chunk_id = ?
        """,
        (
            features.get("f0_min"),
            features.get("f0_max"),
            features.get("f0_mean"),
            features.get("f0_std"),
            features.get("rate_syl"),
            features.get("vcd2tot_frames"),
            features.get("int_min"),
            features.get("int_max"),
            features.get("int_mean"),
            features.get("int_std"),
            features.get("jitter"),
            features.get("shimmer"),
            features.get("nhr"),
            chunk_id,
        ),
    )


def sanitize_chunk_id(chunk_id: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in chunk_id)


def extract_one(
    audio_path: Path,
    praat_script: Path,
    tmp_dir: Path,
    chunk_id: str,
    start_s: float,
    end_s: float,
    text: str,
) -> Dict[str, float]:
    tmp_stem = sanitize_chunk_id(chunk_id)
    cut_wav = tmp_dir / f"{tmp_stem}.wav"
    out_txt = tmp_dir / f"{tmp_stem}.txt"

    try:
        subprocess.check_call([
            "sox",
            str(audio_path),
            str(cut_wav),
            "trim",
            str(start_s),
            "=" + str(end_s),
        ])
        subprocess.check_call([
            "praat",
            "--run",
            str(praat_script),
            str(cut_wav),
            str(out_txt),
        ])

        features = read_praat_output(out_txt)
        duration = end_s - start_s
        features["rate_syl"] = count_syllables(text) / duration if duration > 0 else None
        return features
    finally:
        if cut_wav.exists():
            cut_wav.unlink()
        if out_txt.exists():
            out_txt.unlink()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    praat_script = Path(args.praat_script)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise FileNotFoundError(f"db not found: {db_path}")
    if not praat_script.exists():
        raise FileNotFoundError(f"praat script not found: {praat_script}")

    ensure_tools()

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        total = 0
        extracted = 0
        skipped_short = 0
        skipped_done = 0

        for chunk_id, session_id, audio_path_str, start_s, end_s, text, pitch_mean in iter_targets(conn, args.session_id):
            total += 1
            duration = float(end_s) - float(start_s)

            if duration < args.min_duration:
                skipped_short += 1
                continue
            if (not args.overwrite) and pitch_mean is not None:
                skipped_done += 1
                continue

            audio_path = Path(audio_path_str)
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"audio missing for session={session_id}, chunk={chunk_id}: {audio_path}"
                )

            features = extract_one(
                audio_path=audio_path,
                praat_script=praat_script,
                tmp_dir=tmp_dir,
                chunk_id=chunk_id,
                start_s=float(start_s),
                end_s=float(end_s),
                text=text or "",
            )
            update_chunk_features(conn, chunk_id, features)
            extracted += 1

        conn.commit()
    finally:
        conn.close()

    print(
        "extraction complete: "
        f"total={total} extracted={extracted} skipped_short={skipped_short} skipped_done={skipped_done}"
    )


if __name__ == "__main__":
    main()
