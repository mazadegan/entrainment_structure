#!/usr/bin/env python3
"""Batch runner for the custom entrainment pipeline."""

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

DEFAULT_MODES = ["lsim", "syn", "lcon", "gsim", "gcon"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, help="Path to SQLite DB")
    p.add_argument(
        "--manifest",
        required=True,
        help="CSV manifest with columns: session_id,audio_path,segments_json,align_path",
    )
    p.add_argument("--x-count", type=int, default=10, help="Baseline x pair count")
    p.add_argument("--seed", type=int, default=0, help="Random seed for pairs")
    p.add_argument(
        "--features",
        default="intensity_mean,pitch_mean,rate_syl",
        help="Comma-separated features for measure/validate",
    )
    p.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="Comma-separated measure modes to run",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="CSV export output dir (if omitted, export is skipped)",
    )
    p.add_argument(
        "--export-tables",
        default="lsim_pairs,lsim_summary,syn_summary,lcon_summary,gsim_summary,gcon_summary",
        help="Comma-separated tables to export",
    )
    p.add_argument("--export-prefix", default="", help="Optional CSV filename prefix")
    p.add_argument("--min-feature-fill", type=float, default=1.0)
    p.add_argument(
        "--require-measures",
        default="lsim,syn,lcon,gsim,gcon",
        help="Measure IDs required by validate.py",
    )
    p.add_argument(
        "--tmp-dir",
        default=None,
        help="Optional tmp dir passed to extract.py",
    )
    p.add_argument(
        "--replace-session",
        action="store_true",
        help="Pass --replace-session to ingestion/pairs/measure",
    )
    p.add_argument("--skip-validate", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def parse_csv(raw: str) -> List[str]:
    return [v.strip() for v in raw.split(",") if v.strip()]


def resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def read_manifest(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        required = ["session_id", "audio_path", "segments_json", "align_path"]
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"manifest missing required column(s): {missing}")
        for row in reader:
            if not row.get("session_id"):
                continue
            rows.append(row)
    if not rows:
        raise ValueError("manifest contains no session rows")
    return rows


def run_cmd(cmd: List[str], dry_run: bool) -> None:
    print("$", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    db_path = Path(args.db).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest_dir = manifest_path.parent

    rows = read_manifest(manifest_path)
    modes = parse_csv(args.modes)
    allowed_modes = set(DEFAULT_MODES)
    bad_modes = [m for m in modes if m not in allowed_modes]
    if bad_modes:
        raise ValueError(f"unsupported mode(s): {bad_modes}")

    for row in rows:
        sid = row["session_id"].strip()
        audio_path = resolve_path(manifest_dir, row["audio_path"].strip())
        seg_path = resolve_path(manifest_dir, row["segments_json"].strip())
        align_raw = row.get("align_path", "").strip()
        align_path = resolve_path(manifest_dir, align_raw) if align_raw else None

        print(f"\n=== SESSION {sid} ===")

        # 1) ingest
        ingest_cmd = [
            sys.executable,
            "custom/ingest.py",
            "--db",
            str(db_path),
            "--session-id",
            sid,
            "--audio-path",
            str(audio_path),
            "--segments-json",
            str(seg_path),
        ]
        if align_path:
            ingest_cmd += ["--align", str(align_path)]
        if args.replace_session:
            ingest_cmd += ["--replace-session"]
        run_cmd(ingest_cmd, args.dry_run)

        # 2) extract
        extract_cmd = [
            sys.executable,
            "custom/extract.py",
            "--db",
            str(db_path),
            "--session-id",
            sid,
        ]
        if args.tmp_dir:
            extract_cmd += ["--tmp-dir", args.tmp_dir]
        run_cmd(extract_cmd, args.dry_run)

        # 3) pairs
        pairs_cmd = [
            sys.executable,
            "custom/pairs.py",
            "--db",
            str(db_path),
            "--session-id",
            sid,
            "--x-count",
            str(args.x_count),
            "--seed",
            str(args.seed),
        ]
        if args.replace_session:
            pairs_cmd += ["--replace-session"]
        run_cmd(pairs_cmd, args.dry_run)

        # 4) measures
        for mode in modes:
            measure_cmd = [
                sys.executable,
                "custom/measure.py",
                "--db",
                str(db_path),
                "--session-id",
                sid,
                "--mode",
                mode,
                "--features",
                args.features,
            ]
            if args.replace_session:
                measure_cmd += ["--replace-session"]
            run_cmd(measure_cmd, args.dry_run)

        # 5) validate
        if not args.skip_validate:
            validate_cmd = [
                sys.executable,
                "custom/validate.py",
                "--db",
                str(db_path),
                "--session-id",
                sid,
                "--features",
                args.features,
                "--min-feature-fill",
                str(args.min_feature_fill),
                "--require-measures",
                args.require_measures,
            ]
            run_cmd(validate_cmd, args.dry_run)

        # 6) export
        if (not args.skip_export) and args.out_dir:
            out_dir = Path(args.out_dir)
            export_cmd = [
                sys.executable,
                "custom/export.py",
                "--db",
                str(db_path),
                "--out-dir",
                str(out_dir),
                "--session-id",
                sid,
                "--tables",
                args.export_tables,
            ]
            if args.export_prefix:
                export_cmd += ["--prefix", args.export_prefix]
            run_cmd(export_cmd, args.dry_run)

    print("\nrun_all complete")


if __name__ == "__main__":
    main()
