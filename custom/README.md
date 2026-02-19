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
- Syllable counting is handled by `custom/syllables.py` (standalone).

## Phase B Step 1: Chunk Pairs

Generate adjacent partner (`p`) and non-adjacent baseline (`x`) pairs:

```bash
python custom/pairs.py \
  --db custom/fake_data/custom.db \
  --session-id abc \
  --x-count 10 \
  --seed 0 \
  --replace-session
```

Notes:
- `p`: nearest previous chunk from a different speaker (one per target, when available).
- `x`: up to `x-count` non-adjacent chunks sampled from the same source speaker as `p`.
- Pairs are stored in `chunk_pairs` with columns:
  - `session_id`
  - `p_or_x`
  - `chunk_id_source`
  - `chunk_id_target`
  - `rid` (0 for `p`, 1..N for `x`)

## Phase B Step 2: Local Similarity (lsim-style)

Compute partner-vs-baseline local similarity from `chunk_pairs`:

```bash
python custom/measure.py \
  --db custom/fake_data/custom.db \
  --session-id abc \
  --features intensity_mean,pitch_mean,rate_syl \
  --replace-session
```

Outputs:
- `lsim_pairs`: per target chunk and feature
  - `sim_p` (adjacent partner similarity)
  - `sim_x_mean` (mean non-adjacent baseline similarity)
  - `lsim = -sim_p / sim_x_mean`
- `lsim_summary`: per session and feature summary statistics

Interpretation:
- Similarity is defined as `-abs(source - target)`.
- `lsim > -1` indicates partner similarity exceeds non-adjacent baseline.

## Phase B Step 3: Synchrony (syn)

Compute turn-level synchrony from adjacent partner pairs (`p` only):

```bash
python custom/measure.py \
  --db custom/fake_data/custom.db \
  --session-id abc \
  --mode syn \
  --features intensity_mean,pitch_mean,rate_syl \
  --replace-session
```

Outputs:
- `syn_summary`: per session and feature
  - `n_pairs`
  - `r_value` (Pearson correlation between source and target feature values)
  - `p_value` (when SciPy is available)
  - `dof = n_pairs - 2`

## Phase B Step 4: Local Convergence (lcon)

Compute convergence over time from adjacent partner pairs (`p` only):

```bash
python custom/measure.py \
  --db custom/fake_data/custom.db \
  --session-id abc \
  --mode lcon \
  --features intensity_mean,pitch_mean,rate_syl \
  --replace-session
```

Outputs:
- `lcon_summary`: per session and feature
  - `n_points`
  - `r_value` (Pearson correlation between `sim_p` and target `start_s`)
  - `p_value` (when SciPy is available)
  - `dof = n_points - 2`

Interpretation:
- Similarity uses `sim_p = -abs(source - target)`.
- Positive `r_value` indicates increasing local similarity over time.

## Phase B Step 5: Global Similarity (gsim-style)

Compute session-level partner-vs-baseline similarity:

```bash
python custom/measure.py \
  --db custom/fake_data/custom.db \
  --session-id abc \
  --mode gsim \
  --features intensity_mean,pitch_mean,rate_syl \
  --replace-session
```

Outputs:
- `gsim_summary`: per session and feature
  - `n_targets`
  - `mean_sim_p`
  - `mean_sim_x`
  - `mean_sim_nrm = -mean_sim_p / mean_sim_x`
  - `t_stat`, `p_value` (paired test over target-level partner vs baseline means)

Interpretation:
- Similarity is defined as `-abs(source - target)`.
- `mean_sim_nrm > -1` indicates higher partner similarity than baseline, on average.

## Phase B Step 6: Global Convergence (gcon-style)

Compute session-level convergence between first and second halves:

```bash
python custom/measure.py \
  --db custom/fake_data/custom.db \
  --session-id abc \
  --mode gcon \
  --features intensity_mean,pitch_mean,rate_syl \
  --replace-session
```

Outputs:
- `gcon_summary`: per session and feature
  - `n_pairs` (speaker-pair count used)
  - `mean_dist_first`, `mean_dist_second`
  - `mean_con = mean_dist_first - mean_dist_second`
  - `t_stat`, `p_value`, `dof` (paired test when possible)

Interpretation:
- Positive `mean_con` indicates convergence (smaller distances in second half).

## Validation

Run end-to-end validation checks for a session:

```bash
python custom/validate.py \
  --db custom/fake_data/custom.db \
  --session-id abc \
  --features intensity_mean,pitch_mean,rate_syl \
  --min-feature-fill 1.0 \
  --require-measures lsim,syn,lcon,gsim,gcon
```

Checks include:
- core ingestion counts (`sessions`, `turns`, `chunks`, `word_alignments`)
- feature fill rates in `chunks`
- pair coverage in `chunk_pairs` (`p` and `x`)
- measure table coverage for requested features

Exit code:
- `0` on pass
- `1` on any failed check

## Export to CSV

Export summary and pair-level measure tables to CSV:

```bash
python custom/export.py \
  --db custom/fake_data/custom.db \
  --out-dir custom/fake_data/exports \
  --session-id abc
```

Optional:
- `--tables lsim_summary,syn_summary,lcon_summary,gsim_summary,gcon_summary`
- `--prefix abc_`

Default exported tables:
- `lsim_pairs`
- `lsim_summary`
- `syn_summary`
- `lcon_summary`
- `gsim_summary`
- `gcon_summary`

## Batch Runner (`run_all.py`)

Run the full pipeline for multiple sessions from a manifest CSV.

Manifest columns:
- `session_id`
- `audio_path`
- `segments_json`
- `align_path`

Example manifest (`custom/fake_data/manifest.csv`):

```csv
session_id,audio_path,segments_json,align_path
abc,abc.wav,abc.json,abc.align
```

Run all steps:

```bash
python custom/run_all.py \
  --db custom/fake_data/custom.db \
  --manifest custom/fake_data/manifest.csv \
  --replace-session \
  --x-count 10 \
  --seed 0 \
  --features intensity_mean,pitch_mean,rate_syl \
  --modes lsim,syn,lcon,gsim,gcon \
  --out-dir custom/fake_data/exports \
  --export-prefix batch_
```

Useful flags:
- `--skip-validate`
- `--skip-export`
- `--dry-run`
- `--tmp-dir /tmp`
