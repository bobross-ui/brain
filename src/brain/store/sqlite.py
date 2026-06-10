import asyncio
import hashlib
import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlite_vec

from brain.embeddings import Embedder
from brain.models import (
    IngestResult,
    Memory,
    MemoryAction,
    MemoryActionKind,
    RetrievedEvidence,
    Scope,
    ScoredMemory,
    SessionInput,
    StoredTurn,
)
from brain.store.base import MemoryStore


_MIGRATION_RE = re.compile(r"^(\d+)_.*\.sql$")
_MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_BASELINE_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _metadata_json(metadata: dict | None) -> str:
    return json.dumps(metadata or {}, separators=(",", ":"), sort_keys=True)


def _json_or_none(value: dict | None) -> str | None:
    if value is None:
        return None
    return _metadata_json(value)


def _fts5_query(text: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return " OR ".join(f'"{token}"' for token in tokens)


def _open_db(path: str) -> sqlite3.Connection:
    db_file = Path(path)
    if db_file.parent != Path("."):
        db_file.parent.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(path, isolation_level=None)
    db.execute("PRAGMA foreign_keys=ON")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.row_factory = sqlite3.Row
    return db


def _assert_fts5_available(db: sqlite3.Connection) -> None:
    try:
        db.execute("CREATE VIRTUAL TABLE temp._brain_fts5_check USING fts5(text)")
        db.execute("DROP TABLE temp._brain_fts5_check")
    except sqlite3.Error as exc:
        raise RuntimeError(
            "SQLite FTS5 support is required for raw-turn retrieval"
        ) from exc


def _render_migration(path: Path, dim: int) -> str:
    template = path.read_text()
    return template.replace("__EMBEDDING_DIM__", str(dim))


def _migration_files() -> list[tuple[int, Path]]:
    migrations: list[tuple[int, Path]] = []
    for path in _MIGRATIONS_DIR.iterdir():
        match = _MIGRATION_RE.match(path.name)
        if match:
            migrations.append((int(match.group(1)), path))

    migrations.sort(key=lambda migration: migration[0])
    versions = [version for version, _ in migrations]
    if len(versions) != len(set(versions)):
        raise RuntimeError("Duplicate SQLite migration version")
    return migrations


def _latest_migration_version() -> int:
    migrations = _migration_files()
    return migrations[-1][0] if migrations else 0


def _has_baseline_schema(db: sqlite3.Connection) -> bool:
    rows = db.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name IN ('memories', 'vec_memories')
        """
    ).fetchall()
    return {row["name"] for row in rows} == {"memories", "vec_memories"}


def _current_schema_version(db: sqlite3.Connection) -> int:
    version = int(db.execute("PRAGMA user_version").fetchone()[0])
    if version == 0 and _has_baseline_schema(db):
        # Existing Brain databases predate migrations. Their v1 tables already
        # exist, so stamp them once and let future migrations run normally.
        db.execute(f"PRAGMA user_version = {_BASELINE_SCHEMA_VERSION}")
        return _BASELINE_SCHEMA_VERSION
    return version


def _apply_migration(
    db: sqlite3.Connection,
    *,
    version: int,
    path: Path,
    dim: int,
) -> None:
    sql = _render_migration(path, dim)
    try:
        db.executescript(
            f"""
            BEGIN;
            {sql}
            PRAGMA user_version = {version};
            COMMIT;
            """
        )
    except sqlite3.Error:
        try:
            db.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise


def _apply_schema(db_path: str, dim: int) -> None:
    db = _open_db(db_path)
    try:
        _assert_fts5_available(db)
        current_version = _current_schema_version(db)
        for version, path in _migration_files():
            if version > current_version:
                _apply_migration(db, version=version, path=path, dim=dim)
                current_version = version
    finally:
        db.close()


def _row_to_memory(row: sqlite3.Row | dict[str, Any]) -> Memory:
    return Memory(
        id=row["id"],
        content=row["content"],
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        namespace=row["namespace"],
        metadata=json.loads(row["metadata"]),
        content_hash=row["content_hash"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_turn(row: sqlite3.Row | dict[str, Any]) -> StoredTurn:
    return StoredTurn(
        id=row["id"],
        session_id=row["session_id"],
        source_turn_id=row["source_turn_id"],
        seq=row["seq"],
        speaker=row["speaker"],
        text=row["text"],
        observed_at=row["observed_at"],
        ingested_at=row["ingested_at"],
        user_id=row["user_id"],
        namespace=row["namespace"],
    )


class SQLiteMemoryStore(MemoryStore):
    def __init__(self, db_path: str, embedder: Embedder):
        self._db_path = db_path
        self._embedder = embedder

    @classmethod
    async def create(cls, db_path: str, embedder: Embedder) -> "SQLiteMemoryStore":
        await asyncio.to_thread(_apply_schema, db_path, embedder.dim)
        return cls(db_path, embedder)

    async def add(
        self,
        content: str,
        scope: Scope,
        metadata: dict | None = None,
    ) -> Memory:
        embedding = await self._embedder.embed(content)
        memory_id = str(uuid.uuid4())
        now = _utc_now()
        content_hash = _content_hash(content)
        metadata_text = _metadata_json(metadata)

        def _work() -> dict[str, Any]:
            db = _open_db(self._db_path)
            try:
                cursor = db.execute(
                    """
                    INSERT INTO memories (
                        id, content, user_id, agent_id, namespace, metadata,
                        content_hash, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        content,
                        scope.user_id,
                        scope.agent_id,
                        scope.namespace,
                        metadata_text,
                        content_hash,
                        now,
                        now,
                    ),
                )
                rowid = cursor.lastrowid
                db.execute(
                    """
                    INSERT INTO vec_memories(rowid, embedding, user_id, namespace)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        rowid,
                        sqlite_vec.serialize_float32(embedding),
                        scope.user_id,
                        scope.namespace,
                    ),
                )
                return {
                    "id": memory_id,
                    "content": content,
                    "user_id": scope.user_id,
                    "agent_id": scope.agent_id,
                    "namespace": scope.namespace,
                    "metadata": metadata_text,
                    "content_hash": content_hash,
                    "created_at": now,
                    "updated_at": now,
                }
            finally:
                db.close()

        return _row_to_memory(await asyncio.to_thread(_work))

    async def ingest_session_turns(
        self,
        session: SessionInput,
        scope: Scope,
    ) -> IngestResult:
        if not session.source_session_id:
            raise ValueError("session.source_session_id is required")

        for turn in session.turns:
            if not turn.source_turn_id:
                raise ValueError("turn.source_turn_id is required")

        created_session_id = str(uuid.uuid4())
        now = _utc_now()
        speaker_roster = _json_or_none(session.speaker_roster)
        metadata_text = _metadata_json(session.metadata)
        prepared_turns = [
            {
                "id": str(uuid.uuid4()),
                "source_turn_id": turn.source_turn_id,
                "seq": index,
                "speaker": turn.speaker,
                "text": turn.text,
                "observed_at": turn.observed_at or session.observed_at,
            }
            for index, turn in enumerate(session.turns)
        ]

        def _work() -> dict[str, Any]:
            db = _open_db(self._db_path)
            try:
                db.execute("BEGIN")
                session_row = db.execute(
                    """
                    SELECT id, source_session_id, extraction_completed_at
                    FROM sessions
                    WHERE user_id = ? AND namespace = ? AND source_session_id = ?
                    """,
                    (scope.user_id, scope.namespace, session.source_session_id),
                ).fetchone()
                if session_row is None:
                    session_id = created_session_id
                    db.execute(
                        """
                        INSERT INTO sessions (
                            id, source_session_id, user_id, agent_id, namespace,
                            observed_at, ingested_at, speaker_roster, metadata
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            session.source_session_id,
                            scope.user_id,
                            scope.agent_id,
                            scope.namespace,
                            session.observed_at,
                            now,
                            speaker_roster,
                            metadata_text,
                        ),
                    )
                    extraction_completed_at = None
                else:
                    session_id = session_row["id"]
                    extraction_completed_at = session_row["extraction_completed_at"]

                turn_ids: list[str] = []
                for prepared in prepared_turns:
                    db.execute(
                        """
                        INSERT INTO turns (
                            id, session_id, source_turn_id, seq, speaker, text,
                            observed_at, ingested_at, user_id, namespace
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(session_id, source_turn_id) DO NOTHING
                        """,
                        (
                            prepared["id"],
                            session_id,
                            prepared["source_turn_id"],
                            prepared["seq"],
                            prepared["speaker"],
                            prepared["text"],
                            prepared["observed_at"],
                            now,
                            scope.user_id,
                            scope.namespace,
                        ),
                    )
                    inserted = db.execute("SELECT changes()").fetchone()[0] == 1
                    turn_row = db.execute(
                        """
                        SELECT rowid, id
                        FROM turns
                        WHERE session_id = ? AND source_turn_id = ?
                        """,
                        (session_id, prepared["source_turn_id"]),
                    ).fetchone()
                    if turn_row is None:
                        raise RuntimeError("Failed to resolve stored turn")
                    if inserted:
                        db.execute(
                            "INSERT INTO fts_turns(rowid, text) VALUES (?, ?)",
                            (turn_row["rowid"], prepared["text"]),
                        )
                    turn_ids.append(turn_row["id"])

                db.execute("COMMIT")
                return {
                    "session_id": session_id,
                    "source_session_id": session.source_session_id,
                    "turn_ids": turn_ids,
                    "extraction_completed_at": extraction_completed_at,
                }
            except Exception:
                try:
                    db.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                db.close()

        result = await asyncio.to_thread(_work)
        return IngestResult(**result)

    async def write_extracted_memories(
        self,
        session_id: str,
        scope: Scope,
        actions: list[MemoryAction],
    ) -> list[Memory]:
        now = _utc_now()
        prepared_actions: list[dict[str, Any]] = []
        for action in actions:
            prepared: dict[str, Any] = {
                "kind": action.kind,
                "target_id": action.target_id,
                "content": action.content,
                "metadata": dict(action.metadata),
                "embedding": None,
                "id": str(uuid.uuid4()),
                "created_at": now,
                "updated_at": now,
            }
            if action.kind in (MemoryActionKind.ADD, MemoryActionKind.UPDATE) and action.content:
                prepared["embedding"] = await self._embedder.embed(action.content)
                prepared["content_hash"] = _content_hash(action.content)
            prepared_actions.append(prepared)

        completed_at = _utc_now()

        def _delete_memory_row(db: sqlite3.Connection, rowid: int) -> None:
            db.execute("DELETE FROM vec_memories WHERE rowid = ?", (rowid,))
            db.execute("DELETE FROM memories WHERE rowid = ?", (rowid,))

        def _work() -> list[dict[str, Any]]:
            db = _open_db(self._db_path)
            try:
                db.execute("BEGIN")
                partials = db.execute(
                    """
                    SELECT rowid
                    FROM memories
                    WHERE user_id = ?
                      AND namespace = ?
                      AND json_extract(metadata, '$.ingest_session_id') = ?
                    """,
                    (scope.user_id, scope.namespace, session_id),
                ).fetchall()
                for partial in partials:
                    _delete_memory_row(db, int(partial["rowid"]))

                returned: list[dict[str, Any]] = []
                for prepared in prepared_actions:
                    kind = prepared["kind"]
                    content = prepared["content"]
                    if kind == MemoryActionKind.ADD and content:
                        metadata = dict(prepared["metadata"])
                        metadata["ingest_session_id"] = session_id
                        metadata_text = _metadata_json(metadata)
                        cursor = db.execute(
                            """
                            INSERT INTO memories (
                                id, content, user_id, agent_id, namespace, metadata,
                                content_hash, created_at, updated_at
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                prepared["id"],
                                content,
                                scope.user_id,
                                scope.agent_id,
                                scope.namespace,
                                metadata_text,
                                prepared["content_hash"],
                                prepared["created_at"],
                                prepared["updated_at"],
                            ),
                        )
                        rowid = cursor.lastrowid
                        db.execute(
                            """
                            INSERT INTO vec_memories(rowid, embedding, user_id, namespace)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                rowid,
                                sqlite_vec.serialize_float32(prepared["embedding"]),
                                scope.user_id,
                                scope.namespace,
                            ),
                        )
                        returned.append(
                            {
                                "id": prepared["id"],
                                "content": content,
                                "user_id": scope.user_id,
                                "agent_id": scope.agent_id,
                                "namespace": scope.namespace,
                                "metadata": metadata_text,
                                "content_hash": prepared["content_hash"],
                                "created_at": prepared["created_at"],
                                "updated_at": prepared["updated_at"],
                            }
                        )
                    elif kind == MemoryActionKind.UPDATE and prepared["target_id"] and content:
                        row = db.execute(
                            """
                            SELECT rowid
                            FROM memories
                            WHERE id = ? AND user_id = ? AND namespace = ?
                            """,
                            (prepared["target_id"], scope.user_id, scope.namespace),
                        ).fetchone()
                        if row is None:
                            continue
                        rowid = int(row["rowid"])
                        db.execute(
                            """
                            UPDATE memories
                            SET content = ?, content_hash = ?, updated_at = ?
                            WHERE rowid = ?
                            """,
                            (
                                content,
                                prepared["content_hash"],
                                prepared["updated_at"],
                                rowid,
                            ),
                        )
                        db.execute(
                            """
                            UPDATE vec_memories
                            SET embedding = ?
                            WHERE rowid = ?
                            """,
                            (
                                sqlite_vec.serialize_float32(prepared["embedding"]),
                                rowid,
                            ),
                        )
                        updated = db.execute(
                            """
                            SELECT id, content, user_id, agent_id, namespace,
                                   metadata, content_hash, created_at, updated_at
                            FROM memories
                            WHERE rowid = ?
                            """,
                            (rowid,),
                        ).fetchone()
                        if updated is not None:
                            returned.append(dict(updated))
                    elif kind == MemoryActionKind.DELETE and prepared["target_id"]:
                        row = db.execute(
                            """
                            SELECT rowid
                            FROM memories
                            WHERE id = ? AND user_id = ? AND namespace = ?
                            """,
                            (prepared["target_id"], scope.user_id, scope.namespace),
                        ).fetchone()
                        if row is not None:
                            _delete_memory_row(db, int(row["rowid"]))

                db.execute(
                    """
                    UPDATE sessions
                    SET extraction_completed_at = ?
                    WHERE id = ? AND user_id = ? AND namespace = ?
                    """,
                    (completed_at, session_id, scope.user_id, scope.namespace),
                )
                db.execute("COMMIT")
                return returned
            except Exception:
                try:
                    db.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                db.close()

        rows = await asyncio.to_thread(_work)
        return [_row_to_memory(row) for row in rows]

    async def search(
        self,
        query: str,
        scope: Scope,
        limit: int = 10,
    ) -> list[ScoredMemory]:
        if limit <= 0:
            return []

        query_embedding = await self._embedder.embed(query)

        def _work() -> list[tuple[dict[str, Any], float]]:
            db = _open_db(self._db_path)
            try:
                vector_rows = db.execute(
                    """
                    SELECT rowid, distance
                    FROM vec_memories
                    WHERE embedding MATCH ?
                      AND k = ?
                      AND user_id = ?
                      AND namespace = ?
                    ORDER BY distance
                    """,
                    (
                        sqlite_vec.serialize_float32(query_embedding),
                        limit,
                        scope.user_id,
                        scope.namespace,
                    ),
                ).fetchall()

                if not vector_rows:
                    return []

                distances = {
                    int(row["rowid"]): float(row["distance"]) for row in vector_rows
                }
                placeholders = ",".join("?" for _ in distances)
                rows = db.execute(
                    f"""
                    SELECT rowid, id, content, user_id, agent_id, namespace,
                           metadata, content_hash, created_at, updated_at
                    FROM memories
                    WHERE rowid IN ({placeholders})
                    """,
                    tuple(distances),
                ).fetchall()

                sorted_rows = sorted(
                    rows,
                    key=lambda row: distances[int(row["rowid"])],
                )
                return [
                    (dict(row), distances[int(row["rowid"])])
                    for row in sorted_rows
                ]
            finally:
                db.close()

        rows = await asyncio.to_thread(_work)
        return [
            ScoredMemory(memory=_row_to_memory(row), score=1.0 - distance)
            for row, distance in rows
        ]

    async def get_turn(self, id: str, scope: Scope) -> StoredTurn | None:
        def _work() -> dict[str, Any] | None:
            db = _open_db(self._db_path)
            try:
                row = db.execute(
                    """
                    SELECT id, session_id, source_turn_id, seq, speaker, text,
                           observed_at, ingested_at, user_id, namespace
                    FROM turns
                    WHERE id = ? AND user_id = ? AND namespace = ?
                    """,
                    (id, scope.user_id, scope.namespace),
                ).fetchone()
                return dict(row) if row is not None else None
            finally:
                db.close()

        row = await asyncio.to_thread(_work)
        return _row_to_turn(row) if row is not None else None

    async def search_turns(
        self,
        query: str,
        scope: Scope,
        limit: int = 10,
    ) -> list[RetrievedEvidence]:
        if limit <= 0:
            return []

        fts_query = _fts5_query(query)
        if not fts_query:
            return []

        def _work() -> list[dict[str, Any]]:
            db = _open_db(self._db_path)
            try:
                rows = db.execute(
                    """
                    SELECT
                        turns.id AS turn_id,
                        turns.source_turn_id,
                        turns.speaker,
                        turns.text,
                        turns.observed_at,
                        sessions.source_session_id,
                        bm25(fts_turns) AS bm25_score
                    FROM fts_turns
                    JOIN turns ON turns.rowid = fts_turns.rowid
                    JOIN sessions ON sessions.id = turns.session_id
                    WHERE fts_turns MATCH ?
                      AND turns.user_id = ?
                      AND turns.namespace = ?
                    ORDER BY bm25_score
                    LIMIT ?
                    """,
                    (fts_query, scope.user_id, scope.namespace, limit),
                ).fetchall()
                return [dict(row) for row in rows]
            finally:
                db.close()

        rows = await asyncio.to_thread(_work)
        return [
            RetrievedEvidence(
                kind="turn",
                content=row["text"],
                score=-float(row["bm25_score"]),
                turn_id=row["turn_id"],
                source_turn_ids=[row["source_turn_id"]],
                source_session_id=row["source_session_id"],
                observed_at=row["observed_at"],
            )
            for row in rows
        ]

    async def get(self, id: str, scope: Scope) -> Memory | None:
        def _work() -> dict[str, Any] | None:
            db = _open_db(self._db_path)
            try:
                row = db.execute(
                    """
                    SELECT id, content, user_id, agent_id, namespace, metadata,
                           content_hash, created_at, updated_at
                    FROM memories
                    WHERE id = ? AND user_id = ? AND namespace = ?
                    """,
                    (id, scope.user_id, scope.namespace),
                ).fetchone()
                return dict(row) if row is not None else None
            finally:
                db.close()

        row = await asyncio.to_thread(_work)
        return _row_to_memory(row) if row is not None else None

    async def delete(self, id: str, scope: Scope) -> bool:
        def _work() -> bool:
            db = _open_db(self._db_path)
            try:
                row = db.execute(
                    """
                    SELECT rowid
                    FROM memories
                    WHERE id = ? AND user_id = ? AND namespace = ?
                    """,
                    (id, scope.user_id, scope.namespace),
                ).fetchone()
                if row is None:
                    return False

                rowid = int(row["rowid"])
                db.execute("DELETE FROM vec_memories WHERE rowid = ?", (rowid,))
                db.execute("DELETE FROM memories WHERE rowid = ?", (rowid,))
                return True
            finally:
                db.close()

        return await asyncio.to_thread(_work)

    async def update(self, id: str, content: str, scope: Scope) -> Memory | None:
        existing = await self.get(id, scope)
        if existing is None:
            return None

        embedding = await self._embedder.embed(content)
        content_hash = _content_hash(content)
        updated_at = _utc_now()

        def _work() -> dict[str, Any] | None:
            db = _open_db(self._db_path)
            try:
                row = db.execute(
                    """
                    SELECT rowid
                    FROM memories
                    WHERE id = ? AND user_id = ? AND namespace = ?
                    """,
                    (id, scope.user_id, scope.namespace),
                ).fetchone()
                if row is None:
                    return None

                rowid = int(row["rowid"])
                db.execute(
                    """
                    UPDATE memories
                    SET content = ?, content_hash = ?, updated_at = ?
                    WHERE rowid = ?
                    """,
                    (content, content_hash, updated_at, rowid),
                )
                db.execute(
                    """
                    UPDATE vec_memories
                    SET embedding = ?
                    WHERE rowid = ?
                    """,
                    (sqlite_vec.serialize_float32(embedding), rowid),
                )
                updated = db.execute(
                    """
                    SELECT id, content, user_id, agent_id, namespace, metadata,
                           content_hash, created_at, updated_at
                    FROM memories
                    WHERE rowid = ?
                    """,
                    (rowid,),
                ).fetchone()
                return dict(updated) if updated is not None else None
            finally:
                db.close()

        row = await asyncio.to_thread(_work)
        return _row_to_memory(row) if row is not None else None
