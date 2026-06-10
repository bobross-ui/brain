import sqlite3
from pathlib import Path

import sqlite_vec

from brain.store.sqlite import (
    _apply_schema,
    _latest_migration_version,
    _open_db,
)


def _table_names(db: sqlite3.Connection) -> set[str]:
    rows = db.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {row["name"] for row in rows}


def _create_baseline_db(
    db_path: Path,
    dim: int,
    *,
    user_version: int = 0,
) -> None:
    db = sqlite3.connect(db_path, isolation_level=None)
    try:
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        db.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS memories (
                id           TEXT PRIMARY KEY,
                content      TEXT NOT NULL,
                user_id      TEXT NOT NULL,
                agent_id     TEXT,
                namespace    TEXT NOT NULL DEFAULT 'default',
                metadata     TEXT NOT NULL DEFAULT '{{}}',
                content_hash TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memories_user_ns
                ON memories (user_id, namespace);

            CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                embedding float[{dim}],
                user_id TEXT PARTITION KEY,
                namespace TEXT PARTITION KEY
            );
            """
        )
        if user_version:
            db.execute(f"PRAGMA user_version = {user_version}")
    finally:
        db.close()


def test_apply_migrations_to_empty_db_sets_latest_version(tmp_path: Path) -> None:
    db_path = tmp_path / "brain.db"

    _apply_schema(str(db_path), 384)

    db = _open_db(str(db_path))
    try:
        assert db.execute("PRAGMA user_version").fetchone()[0] == (
            _latest_migration_version()
        )
        assert {
            "memories",
            "vec_memories",
            "sessions",
            "turns",
            "fts_turns",
            "fts_memories",
            "memory_sources",
        }.issubset(_table_names(db))
        memory_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(memories)").fetchall()
        }
        assert {"subject", "source_session_id", "observed_at"}.issubset(
            memory_columns
        )
    finally:
        db.close()


def test_apply_migrations_to_v1_db_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "brain.db"
    _create_baseline_db(db_path, 384, user_version=1)

    _apply_schema(str(db_path), 384)
    _apply_schema(str(db_path), 384)

    db = _open_db(str(db_path))
    try:
        assert db.execute("PRAGMA user_version").fetchone()[0] == (
            _latest_migration_version()
        )
        assert {
            "memories",
            "vec_memories",
            "sessions",
            "turns",
            "fts_turns",
            "fts_memories",
            "memory_sources",
        }.issubset(_table_names(db))
    finally:
        db.close()


def test_implicit_v0_baseline_schema_is_stamped_to_v1(tmp_path: Path) -> None:
    db_path = tmp_path / "brain.db"
    _create_baseline_db(db_path, 384)

    _apply_schema(str(db_path), 384)

    db = _open_db(str(db_path))
    try:
        assert db.execute("PRAGMA user_version").fetchone()[0] == (
            _latest_migration_version()
        )
    finally:
        db.close()


def test_migrations_preserve_existing_v1_memory_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "brain.db"
    _create_baseline_db(db_path, 384, user_version=1)
    db = _open_db(str(db_path))
    try:
        db.execute(
            """
            INSERT INTO memories (
                id, content, user_id, agent_id, namespace, metadata,
                content_hash, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "memory-1",
                "Alice likes pasta.",
                "alice",
                None,
                "default",
                "{}",
                "hash",
                "2026-06-09T00:00:00+00:00",
                "2026-06-09T00:00:00+00:00",
            ),
        )
    finally:
        db.close()

    _apply_schema(str(db_path), 384)

    db = _open_db(str(db_path))
    try:
        row = db.execute(
            "SELECT content FROM memories WHERE id = ?",
            ("memory-1",),
        ).fetchone()
        assert row["content"] == "Alice likes pasta."
        fts_row = db.execute(
            """
            SELECT rowid
            FROM fts_memories
            WHERE fts_memories MATCH ?
            """,
            ('"pasta"',),
        ).fetchone()
        assert fts_row is not None
        assert db.execute("PRAGMA user_version").fetchone()[0] == (
            _latest_migration_version()
        )
    finally:
        db.close()


def test_open_db_enables_foreign_keys(tmp_path: Path) -> None:
    db = _open_db(str(tmp_path / "brain.db"))
    try:
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        db.close()
