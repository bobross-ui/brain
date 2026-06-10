CREATE VIRTUAL TABLE IF NOT EXISTS fts_memories USING fts5(
    content,
    subject,
    content='memories',
    content_rowid='rowid'
);

INSERT INTO fts_memories(fts_memories) VALUES('rebuild');
