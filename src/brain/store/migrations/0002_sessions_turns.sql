CREATE TABLE IF NOT EXISTS sessions (
    id                       TEXT PRIMARY KEY,
    source_session_id        TEXT NOT NULL,
    user_id                  TEXT NOT NULL,
    agent_id                 TEXT,
    namespace                TEXT NOT NULL DEFAULT 'default',
    observed_at              TEXT,
    ingested_at              TEXT NOT NULL,
    extraction_completed_at  TEXT,
    speaker_roster           TEXT,
    metadata                 TEXT NOT NULL DEFAULT '{}',
    UNIQUE(user_id, namespace, source_session_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_ns
    ON sessions (user_id, namespace);

CREATE TABLE IF NOT EXISTS turns (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source_turn_id   TEXT NOT NULL,
    seq              INTEGER NOT NULL,
    speaker          TEXT NOT NULL,
    text             TEXT NOT NULL,
    observed_at      TEXT,
    ingested_at      TEXT NOT NULL,
    user_id          TEXT NOT NULL,
    namespace        TEXT NOT NULL,
    UNIQUE(session_id, source_turn_id)
);

CREATE INDEX IF NOT EXISTS idx_turns_session
    ON turns (session_id, seq);

CREATE INDEX IF NOT EXISTS idx_turns_user_ns
    ON turns (user_id, namespace);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_turns USING fts5(
    text,
    content='turns',
    content_rowid='rowid'
);
