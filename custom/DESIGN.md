# Custom Ingestion Pipeline Design (Full Compatibility)

## Goal
Build a custom ingestion and extraction pipeline that is as close as possible to the existing repository workflow, while avoiding corpus-specific input formatting requirements.

## Recommended Decisions

### 1. Compatibility Level
- Choose **full-repo-compatible** (not feature-only).
- Rationale: maximizes alignment with existing SQL + `ap.py` entrainment measures.

### 2. Canonical Internal Schema
Use a single internal schema for all sources:
- `session_id`
- `speaker_id`
- `turn_id`
- `chunk_id`
- `audio_path`
- `start_s`
- `end_s`
- `text`
- `words` (list with word-level times)

Time conventions:
- Keep all times in **seconds** as **float** values everywhere.

### 3. Segmentation Policy
- Start with **JSON segments as chunks** (fastest and lowest-risk path).
- Add optional later mode to split chunks by pause threshold:
  - split when pause `>= 0.05s` (paper-style IPU behavior).

### 4. `.align` Format
Finalize `.align` format as:
- `segment_id word_index start_s_rel end_s_rel word`

Constraint:
- Do **not** use resetting timelines without `segment_id`.

### 5. Output Store
- Use **SQLite** with repo-like tables:
  - `sessions`
  - `turns`
  - `chunks`
  - feature columns in `chunks` (pitch/intensity/jitter/shimmer/nhr/rates)

Rationale:
- Preserves compatibility with existing SQL flow and `ap.py`.

## MVP Delivery Phases

### Phase A
- Ingestion
- Feature extraction
- Quality control (QC)

### Phase B
- Chunk-pair generation (adjacent + non-adjacent baseline)
- Entrainment measure computation

## Notes
- This document intentionally defines architecture and data contracts before implementation.
- Immediate implementation will follow these decisions under the `custom/` directory.
