#!/usr/bin/env python3
"""Generate adjacent ('p') and non-adjacent baseline ('x') chunk pairs."""

import argparse
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Chunk:
    chunk_id: str
    session_id: str
    speaker_id: str
    start_s: float
    end_s: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, help="Path to SQLite DB")
    p.add_argument("--session-id", default=None, help="Restrict to one session")
    p.add_argument(
        "--x-count",
        type=int,
        default=10,
        help="Max number of non-adjacent baseline pairs per target",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for baseline sampling")
    p.add_argument(
        "--replace-session",
        action="store_true",
        help="Delete existing pairs for targeted session(s) before insertion",
    )
    return p.parse_args()


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_pairs (
            pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            p_or_x TEXT NOT NULL CHECK (p_or_x IN ('p', 'x')),
            chunk_id_source TEXT NOT NULL,
            chunk_id_target TEXT NOT NULL,
            rid INTEGER NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
            FOREIGN KEY (chunk_id_source) REFERENCES chunks (chunk_id) ON DELETE CASCADE,
            FOREIGN KEY (chunk_id_target) REFERENCES chunks (chunk_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_chunk_pairs_unique
        ON chunk_pairs (session_id, p_or_x, chunk_id_source, chunk_id_target, rid)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunk_pairs_target
        ON chunk_pairs (session_id, chunk_id_target)
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


def load_chunks(conn: sqlite3.Connection, session_id: str) -> List[Chunk]:
    rows = conn.execute(
        """
        SELECT chunk_id, session_id, speaker_id, start_s, end_s
        FROM chunks
        WHERE session_id = ?
        ORDER BY start_s, end_s, chunk_id
        """,
        (session_id,),
    ).fetchall()
    return [Chunk(*r) for r in rows]


def find_adjacent_source(target_idx: int, chunks: List[Chunk]) -> Optional[int]:
    """Find nearest previous chunk from a different speaker for target index."""
    target = chunks[target_idx]
    for i in range(target_idx - 1, -1, -1):
        if chunks[i].speaker_id != target.speaker_id:
            return i
    return None


def build_pairs_for_session(
    chunks: List[Chunk], x_count: int, rng: random.Random
) -> List[tuple]:
    pairs: List[tuple] = []

    # Pools by speaker for fast baseline sampling.
    by_speaker: Dict[str, List[int]] = {}
    for idx, c in enumerate(chunks):
        by_speaker.setdefault(c.speaker_id, []).append(idx)

    for t_idx, target in enumerate(chunks):
        src_idx = find_adjacent_source(t_idx, chunks)
        if src_idx is None:
            continue

        src = chunks[src_idx]
        # adjacent partner pair
        pairs.append((target.session_id, "p", src.chunk_id, target.chunk_id, 0))

        # non-adjacent baseline pairs from same source speaker
        candidate_idxs = [
            i
            for i in by_speaker.get(src.speaker_id, [])
            if i != src_idx and chunks[i].chunk_id != target.chunk_id
        ]
        if not candidate_idxs:
            continue

        k = min(x_count, len(candidate_idxs))
        sampled = rng.sample(candidate_idxs, k=k)
        sampled.sort(key=lambda i: chunks[i].start_s)
        for rid, i in enumerate(sampled, start=1):
            x_src = chunks[i]
            pairs.append((target.session_id, "x", x_src.chunk_id, target.chunk_id, rid))

    return pairs


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"db not found: {db_path}")

    rng = random.Random(args.seed)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        ensure_schema(conn)

        session_ids = get_session_ids(conn, args.session_id)
        if not session_ids:
            raise ValueError("no matching sessions found")

        total_pairs = 0
        session_stats: List[tuple] = []

        for sid in session_ids:
            if args.replace_session:
                conn.execute("DELETE FROM chunk_pairs WHERE session_id = ?", (sid,))

            chunks = load_chunks(conn, sid)
            pairs = build_pairs_for_session(chunks, args.x_count, rng)
            if not pairs:
                session_stats.append((sid, 0, 0))
                continue

            conn.executemany(
                """
                INSERT OR REPLACE INTO chunk_pairs
                (session_id, p_or_x, chunk_id_source, chunk_id_target, rid)
                VALUES (?, ?, ?, ?, ?)
                """,
                pairs,
            )

            p_cnt = sum(1 for p in pairs if p[1] == "p")
            x_cnt = sum(1 for p in pairs if p[1] == "x")
            session_stats.append((sid, p_cnt, x_cnt))
            total_pairs += len(pairs)

        conn.commit()
    finally:
        conn.close()

    summary = " ".join([f"{sid}[p={p},x={x}]" for sid, p, x in session_stats])
    print(f"pair generation complete: sessions={len(session_stats)} total_pairs={total_pairs} {summary}")


if __name__ == "__main__":
    main()
