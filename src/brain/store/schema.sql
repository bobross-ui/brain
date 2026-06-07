CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    content      TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    agent_id     TEXT,
    namespace    TEXT NOT NULL DEFAULT 'default',
    metadata     TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_user_ns ON memories (user_id, namespace);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
    embedding float[384],
    user_id TEXT PARTITION KEY,
    namespace TEXT PARTITION KEY
);
