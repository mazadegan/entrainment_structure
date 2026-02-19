#!/usr/bin/env python3
"""Export custom pipeline SQLite tables to CSV files."""

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import List

DEFAULT_TABLES = [
    "lsim_pairs",
    "lsim_summary",
    "syn_summary",
    "lcon_summary",
    "gsim_summary",
    "gcon_summary",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, help="Path to SQLite DB")
    p.add_argument("--out-dir", required=True, help="Output directory for CSV files")
    p.add_argument("--session-id", default=None, help="Optional session filter")
    p.add_argument(
        "--tables",
        default=",".join(DEFAULT_TABLES),
        help="Comma-separated table names to export",
    )
    p.add_argument(
        "--prefix",
        default="",
        help="Optional filename prefix (e.g. run1_)",
    )
    return p.parse_args()


def parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def export_table(
    conn: sqlite3.Connection,
    table: str,
    out_path: Path,
    session_id: str = None,
) -> int:
    if session_id:
        sql = f"SELECT * FROM {table} WHERE session_id = ?"
        rows = conn.execute(sql, (session_id,)).fetchall()
    else:
        sql = f"SELECT * FROM {table}"
        rows = conn.execute(sql).fetchall()

    colnames = [d[0] for d in conn.execute(sql + (" LIMIT 0" if not session_id else " LIMIT 0"), (session_id,) if session_id else ()).description]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(colnames)
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    tables = parse_csv(args.tables)

    if not db_path.exists():
        raise FileNotFoundError(f"db not found: {db_path}")
    if not tables:
        raise ValueError("no tables specified")

    conn = sqlite3.connect(str(db_path))
    try:
        exported = 0
        for table in tables:
            if not table_exists(conn, table):
                print(f"SKIP: missing table {table}")
                continue

            fname = f"{args.prefix}{table}.csv"
            n = export_table(conn, table, out_dir / fname, args.session_id)
            exported += 1
            print(f"WROTE: {out_dir / fname} rows={n}")

        print(f"export complete: files={exported} out_dir={out_dir}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
