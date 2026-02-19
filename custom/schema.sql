PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    audio_path TEXT NOT NULL,
    segments_json_path TEXT NOT NULL,
    align_path TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    speaker_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    speaker_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    duration REAL NOT NULL,
    text TEXT NOT NULL,
    words_json TEXT,
    pitch_min NUMERIC,
    pitch_max NUMERIC,
    pitch_mean NUMERIC,
    pitch_std NUMERIC,
    rate_syl NUMERIC,
    rate_vcd NUMERIC,
    intensity_min NUMERIC,
    intensity_max NUMERIC,
    intensity_mean NUMERIC,
    intensity_std NUMERIC,
    jitter NUMERIC,
    shimmer NUMERIC,
    nhr NUMERIC,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
    FOREIGN KEY (turn_id) REFERENCES turns (turn_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS word_alignments (
    session_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    word_index INTEGER NOT NULL,
    word TEXT NOT NULL,
    start_s_rel REAL NOT NULL,
    end_s_rel REAL NOT NULL,
    start_s_abs REAL NOT NULL,
    end_s_abs REAL NOT NULL,
    PRIMARY KEY (session_id, chunk_id, word_index),
    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
    FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns (session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks (session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_turn ON chunks (turn_id);
CREATE INDEX IF NOT EXISTS idx_word_align_chunk ON word_alignments (chunk_id);
