# Custom Pipeline (Phase A Ingestion)

## Files
- `custom/DESIGN.md`: design decisions
- `custom/schema.sql`: SQLite schema for custom ingestion
- `custom/ingest.py`: CLI to ingest segment metadata + optional word alignments

## CLI Usage

```bash
python custom/ingest.py \
  --db custom/custom.db \
  --session-id sess001 \
  --audio-path /path/to/audio.wav \
  --segments-json /path/to/segments.json \
  --align /path/to/words.align \
  --replace-session
```

## Required JSON Shape
`segments.json` must include a top-level `segments` list. Each segment must have:
- `tier`
- `index`
- `start_ms`
- `end_ms`
- `transcript`

## `.align` Format
Whitespace-delimited, one word per line:

```text
segment_id word_index start_s_rel end_s_rel word
```

Example:

```text
speaker1:1 1 0.00 0.35 out
speaker1:1 2 0.35 0.55 are
speaker1:1 3 0.55 0.90 just
```

Where `segment_id` must match `tier:index` from the JSON segment.

## What Gets Written
- `sessions`
- `turns`
- `chunks`
- `word_alignments`

Current behavior:
- Each JSON segment is ingested as one turn and one chunk.
- Feature columns in `chunks` are present and initialized as `NULL` for Phase A extraction to fill later.

## Phase A Extraction

Populate chunk feature columns from source audio:

```bash
python custom/extract.py \
  --db custom/fake_data/custom.db \
  --session-id abc
```

Optional flags:
- `--overwrite`: recompute features for chunks already populated
- `--min-duration 0.04`: minimum chunk length in seconds
- `--tmp-dir /tmp`: temporary working directory
- `--praat-script praat/extract_features.praat`: custom praat script path

Requirements:
- `sox` available on `PATH`
- `praat` available on `PATH`
