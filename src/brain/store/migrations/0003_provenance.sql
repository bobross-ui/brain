ALTER TABLE memories ADD COLUMN subject TEXT;
ALTER TABLE memories ADD COLUMN source_session_id TEXT;
ALTER TABLE memories ADD COLUMN observed_at TEXT;

CREATE TABLE memory_sources (
    memory_id  TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    turn_id    TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, turn_id)
);

CREATE INDEX idx_memory_sources_turn
    ON memory_sources (turn_id);
