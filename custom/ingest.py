#!/usr/bin/env python3
"""Ingest JSON segment metadata + optional .align into custom SQLite schema."""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, help="Path to SQLite DB file")
    p.add_argument("--session-id", required=True, help="Stable session identifier")
    p.add_argument("--audio-path", required=True, help="Absolute/relative path to source wav")
    p.add_argument("--segments-json", required=True, help="JSON file containing 'segments'")
    p.add_argument(
        "--align",
        default=None,
        help="Optional .align file (segment_id word_index start_s_rel end_s_rel word)",
    )
    p.add_argument(
        "--schema",
        default=str(Path(__file__).with_name("schema.sql")),
        help="Path to SQL schema file (default: custom/schema.sql)",
    )
    p.add_argument(
        "--replace-session",
        action="store_true",
        help="Delete existing rows for this session_id before inserting",
    )
    return p.parse_args()


def load_segments(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments-json must contain a non-empty 'segments' list")
    required = ["tier", "index", "start_ms", "end_ms", "transcript"]
    for i, seg in enumerate(segments, start=1):
        missing = [k for k in required if k not in seg]
        if missing:
            raise ValueError(f"segment #{i} missing fields: {missing}")
    return segments


def parse_align(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for ln, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=4)
        if len(parts) != 5:
            raise ValueError(
                f"align parse error line {ln}: expected 5 columns "
                "(segment_id word_index start_s_rel end_s_rel word)"
            )
        segment_id, widx_s, start_s, end_s, word = parts
        try:
            word_index = int(widx_s)
            start_rel = float(start_s)
            end_rel = float(end_s)
        except ValueError as exc:
            raise ValueError(f"align parse error line {ln}: {exc}") from exc
        if end_rel < start_rel:
            raise ValueError(f"align line {ln}: end_s_rel < start_s_rel")
        out.setdefault(segment_id, []).append(
            {
                "word_index": word_index,
                "word": word,
                "start_s_rel": start_rel,
                "end_s_rel": end_rel,
            }
        )
    for segment_id, items in out.items():
        items.sort(key=lambda x: x["word_index"])
        seen = set()
        for item in items:
            idx = item["word_index"]
            if idx in seen:
                raise ValueError(f"duplicate word_index {idx} in segment_id={segment_id}")
            seen.add(idx)
    return out


def segment_key(segment: Dict[str, Any]) -> str:
    return f"{segment['tier']}:{segment['index']}"


def normalize_rows(
    session_id: str,
    audio_path: str,
    segments: List[Dict[str, Any]],
    align_map: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[Tuple], List[Tuple], List[Tuple], List[Tuple]]:
    session_row = [(session_id, audio_path)]

    turns: List[Tuple] = []
    chunks: List[Tuple] = []
    word_rows: List[Tuple] = []

    # Per-speaker running turn index; for now each segment is one turn/chunk.
    turn_counter: Dict[str, int] = {}

    ordered = sorted(segments, key=lambda s: (float(s["start_ms"]), float(s["end_ms"])))
    for seg in ordered:
        spk = str(seg["tier"])
        turn_counter.setdefault(spk, 0)
        turn_counter[spk] += 1

        start_s = float(seg["start_ms"]) / 1000.0
        end_s = float(seg["end_ms"]) / 1000.0
        if end_s <= start_s:
            raise ValueError(f"segment {segment_key(seg)} has non-positive duration")

        key = segment_key(seg)
        turn_id = f"{session_id}:{key}:turn"
        chunk_id = f"{session_id}:{key}:chunk"
        text = str(seg.get("transcript", "")).strip()

        align_words = align_map.get(key, [])
        words_payload = [
            {
                "word_index": w["word_index"],
                "word": w["word"],
                "start_s_rel": w["start_s_rel"],
                "end_s_rel": w["end_s_rel"],
            }
            for w in align_words
        ]

        turns.append((turn_id, session_id, spk, turn_counter[spk], start_s, end_s))
        chunks.append(
            (
                chunk_id,
                session_id,
                turn_id,
                spk,
                1,
                start_s,
                end_s,
                end_s - start_s,
                text,
                json.dumps(words_payload),
            )
        )

        for w in align_words:
            word_rows.append(
                (
                    session_id,
                    chunk_id,
                    w["word_index"],
                    w["word"],
                    w["start_s_rel"],
                    w["end_s_rel"],
                    start_s + w["start_s_rel"],
                    start_s + w["end_s_rel"],
                )
            )

    # Validate align coverage and unknown segment_ids.
    unknown_ids = sorted(set(align_map.keys()) - {segment_key(s) for s in segments})
    if unknown_ids:
        head = ", ".join(unknown_ids[:5])
        raise ValueError(f"align contains unknown segment_id(s), e.g. {head}")

    return session_row, turns, chunks, word_rows


def init_schema(conn: sqlite3.Connection, schema_path: Path) -> None:
    sql = schema_path.read_text()
    conn.executescript(sql)


def upsert_session(
    conn: sqlite3.Connection,
    session_id: str,
    audio_path: str,
    segments_json_path: str,
    align_path: str,
    turns: List[Tuple],
    chunks: List[Tuple],
    word_rows: List[Tuple],
    replace_session: bool,
) -> None:
    cur = conn.cursor()

    if replace_session:
        cur.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    else:
        cur.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
        if cur.fetchone() is not None:
            raise ValueError(
                f"session_id '{session_id}' already exists; use --replace-session to overwrite"
            )

    cur.execute(
        """
        INSERT INTO sessions (session_id, audio_path, segments_json_path, align_path)
        VALUES (?, ?, ?, ?)
        """,
        (session_id, audio_path, segments_json_path, align_path),
    )

    cur.executemany(
        """
        INSERT INTO turns (turn_id, session_id, speaker_id, turn_index, start_s, end_s)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        turns,
    )

    cur.executemany(
        """
        INSERT INTO chunks (
            chunk_id, session_id, turn_id, speaker_id, chunk_index,
            start_s, end_s, duration, text, words_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        chunks,
    )

    if word_rows:
        cur.executemany(
            """
            INSERT INTO word_alignments (
                session_id, chunk_id, word_index, word,
                start_s_rel, end_s_rel, start_s_abs, end_s_abs
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            word_rows,
        )



def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    schema_path = Path(args.schema)
    segments_path = Path(args.segments_json)
    align_path = Path(args.align) if args.align else None

    segments = load_segments(segments_path)
    align_map = parse_align(align_path) if align_path else {}

    _, turns, chunks, word_rows = normalize_rows(
        session_id=args.session_id,
        audio_path=args.audio_path,
        segments=segments,
        align_map=align_map,
    )

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        init_schema(conn, schema_path)
        upsert_session(
            conn,
            session_id=args.session_id,
            audio_path=args.audio_path,
            segments_json_path=str(segments_path),
            align_path=str(align_path) if align_path else None,
            turns=turns,
            chunks=chunks,
            word_rows=word_rows,
            replace_session=args.replace_session,
        )
        conn.commit()
    finally:
        conn.close()

    print(
        "ingestion complete: "
        f"session_id={args.session_id} chunks={len(chunks)} words={len(word_rows)} db={db_path}"
    )


if __name__ == "__main__":
    main()
