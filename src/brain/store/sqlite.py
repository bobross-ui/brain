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
from brain.retrieval import (
    FilterSpec,
    reciprocal_rank_fusion,
    search_pool_limit,
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


def _insert_fts_memory(
    db: sqlite3.Connection,
    rowid: int,
    content: str,
    subject: str | None,
) -> None:
    db.execute(
        """
        INSERT INTO fts_memories(rowid, content, subject)
        VALUES (?, ?, ?)
        """,
        (rowid, content, subject),
    )


def _delete_fts_memory(
    db: sqlite3.Connection,
    rowid: int,
    content: str,
    subject: str | None,
) -> None:
    db.execute(
        """
        INSERT INTO fts_memories(fts_memories, rowid, content, subject)
        VALUES ('delete', ?, ?, ?)
        """,
        (rowid, content, subject),
    )


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
            "SQLite FTS5 support is required for hybrid retrieval"
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


def _memory_select_columns(alias: str = "memories") -> str:
    return f"""
        {alias}.id,
        {alias}.content,
        {alias}.user_id,
        {alias}.agent_id,
        {alias}.namespace,
        {alias}.metadata,
        {alias}.subject,
        {alias}.source_session_id,
        {alias}.observed_at,
        {alias}.content_hash,
        {alias}.created_at,
        {alias}.updated_at,
        COALESCE(
            (
                SELECT json_group_array(linked_turns.source_turn_id)
                FROM (
                    SELECT turns.source_turn_id
                    FROM memory_sources
                    JOIN turns ON turns.id = memory_sources.turn_id
                    WHERE memory_sources.memory_id = {alias}.id
                    ORDER BY turns.seq, turns.source_turn_id
                ) AS linked_turns
            ),
            '[]'
        ) AS source_turn_ids
    """


def _row_to_memory(row: sqlite3.Row | dict[str, Any]) -> Memory:
    return Memory(
        id=row["id"],
        content=row["content"],
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        namespace=row["namespace"],
        metadata=json.loads(row["metadata"]),
        subject=row["subject"],
        source_turn_ids=json.loads(row["source_turn_ids"]),
        source_session_id=row["source_session_id"],
        observed_at=row["observed_at"],
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
    async def create(
        cls,
        db_path: str,
        embedder: Embedder,
    ) -> "SQLiteMemoryStore":
        await asyncio.to_thread(_apply_schema, db_path, embedder.dim)
        return cls(db_path, embedder)

    async def add(
        self,
        content: str,
        scope: Scope,
        metadata: dict | None = None,
        *,
        subject: str | None = None,
        internal_turn_ids: list[str] | None = None,
        observed_at: str | None = None,
        source_session_id: str | None = None,
    ) -> Memory:
        embedding = await self._embedder.embed(content)
        memory_id = str(uuid.uuid4())
        now = _utc_now()
        content_hash = _content_hash(content)
        metadata_text = _metadata_json(metadata)

        def _work() -> dict[str, Any]:
            db = _open_db(self._db_path)
            try:
                db.execute("BEGIN")
                cursor = db.execute(
                    """
                    INSERT INTO memories (
                        id, content, user_id, agent_id, namespace, metadata,
                        subject, source_session_id, observed_at, content_hash,
                        created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        content,
                        scope.user_id,
                        scope.agent_id,
                        scope.namespace,
                        metadata_text,
                        subject,
                        source_session_id,
                        observed_at,
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
                _insert_fts_memory(db, int(rowid), content, subject)
                for turn_id in dict.fromkeys(internal_turn_ids or []):
                    db.execute(
                        """
                        INSERT INTO memory_sources(memory_id, turn_id)
                        VALUES (?, ?)
                        """,
                        (memory_id, turn_id),
                    )
                row = db.execute(
                    f"""
                    SELECT {_memory_select_columns("memories")}
                    FROM memories
                    WHERE memories.id = ?
                    """,
                    (memory_id,),
                ).fetchone()
                db.execute("COMMIT")
                if row is None:
                    raise RuntimeError("Failed to read inserted memory")
                return dict(row)
            except Exception:
                try:
                    db.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
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
                "subject": action.subject,
                "internal_turn_ids": action.internal_turn_ids,
                "source_session_id": action.source_session_id,
                "observed_at": action.observed_at,
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

        def _delete_memory_row(
            db: sqlite3.Connection,
            rowid: int,
            content: str,
            subject: str | None,
        ) -> None:
            _delete_fts_memory(db, rowid, content, subject)
            db.execute("DELETE FROM vec_memories WHERE rowid = ?", (rowid,))
            db.execute("DELETE FROM memories WHERE rowid = ?", (rowid,))

        def _work() -> list[dict[str, Any]]:
            db = _open_db(self._db_path)
            try:
                db.execute("BEGIN")
                partials = db.execute(
                    """
                    SELECT rowid, content, subject
                    FROM memories
                    WHERE user_id = ?
                      AND namespace = ?
                      AND json_extract(metadata, '$.ingest_session_id') = ?
                    """,
                    (scope.user_id, scope.namespace, session_id),
                ).fetchall()
                for partial in partials:
                    _delete_memory_row(
                        db,
                        int(partial["rowid"]),
                        partial["content"],
                        partial["subject"],
                    )

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
                                subject, source_session_id, observed_at, content_hash,
                                created_at, updated_at
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                prepared["id"],
                                content,
                                scope.user_id,
                                scope.agent_id,
                                scope.namespace,
                                metadata_text,
                                prepared["subject"],
                                prepared["source_session_id"],
                                prepared["observed_at"],
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
                        _insert_fts_memory(
                            db,
                            int(rowid),
                            content,
                            prepared["subject"],
                        )
                        for turn_id in dict.fromkeys(
                            prepared["internal_turn_ids"] or []
                        ):
                            db.execute(
                                """
                                INSERT INTO memory_sources(memory_id, turn_id)
                                VALUES (?, ?)
                                """,
                                (prepared["id"], turn_id),
                            )
                        inserted = db.execute(
                            f"""
                            SELECT {_memory_select_columns("memories")}
                            FROM memories
                            WHERE memories.id = ?
                            """,
                            (prepared["id"],),
                        ).fetchone()
                        if inserted is not None:
                            returned.append(dict(inserted))
                    elif kind == MemoryActionKind.UPDATE and prepared["target_id"] and content:
                        row = db.execute(
                            """
                            SELECT rowid, content, subject, metadata
                            FROM memories
                            WHERE id = ? AND user_id = ? AND namespace = ?
                            """,
                            (prepared["target_id"], scope.user_id, scope.namespace),
                        ).fetchone()
                        if row is None:
                            continue
                        rowid = int(row["rowid"])
                        metadata = json.loads(row["metadata"])
                        metadata.update(prepared["metadata"])
                        next_subject = (
                            prepared["subject"]
                            if prepared["subject"] is not None
                            else row["subject"]
                        )
                        _delete_fts_memory(
                            db,
                            rowid,
                            row["content"],
                            row["subject"],
                        )
                        db.execute(
                            """
                            UPDATE memories
                            SET content = ?,
                                metadata = ?,
                                subject = COALESCE(?, subject),
                                source_session_id = COALESCE(?, source_session_id),
                                observed_at = COALESCE(?, observed_at),
                                content_hash = ?,
                                updated_at = ?
                            WHERE rowid = ?
                            """,
                            (
                                content,
                                _metadata_json(metadata),
                                prepared["subject"],
                                prepared["source_session_id"],
                                prepared["observed_at"],
                                prepared["content_hash"],
                                prepared["updated_at"],
                                rowid,
                            ),
                        )
                        _insert_fts_memory(db, rowid, content, next_subject)
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
                        if prepared["internal_turn_ids"] is not None:
                            db.execute(
                                "DELETE FROM memory_sources WHERE memory_id = ?",
                                (prepared["target_id"],),
                            )
                            for turn_id in dict.fromkeys(
                                prepared["internal_turn_ids"]
                            ):
                                db.execute(
                                    """
                                    INSERT INTO memory_sources(memory_id, turn_id)
                                    VALUES (?, ?)
                                    """,
                                    (prepared["target_id"], turn_id),
                                )
                        updated = db.execute(
                            f"""
                            SELECT {_memory_select_columns("memories")}
                            FROM memories
                            WHERE memories.rowid = ?
                            """,
                            (rowid,),
                        ).fetchone()
                        if updated is not None:
                            returned.append(dict(updated))
                    elif kind == MemoryActionKind.DELETE and prepared["target_id"]:
                        row = db.execute(
                            """
                            SELECT rowid, content, subject
                            FROM memories
                            WHERE id = ? AND user_id = ? AND namespace = ?
                            """,
                            (prepared["target_id"], scope.user_id, scope.namespace),
                        ).fetchone()
                        if row is not None:
                            _delete_memory_row(
                                db,
                                int(row["rowid"]),
                                row["content"],
                                row["subject"],
                            )

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
        *,
        filters: dict | None = None,
        mode: str = "hybrid",
    ) -> list[ScoredMemory]:
        if limit <= 0:
            return []

        if mode not in {"hybrid", "vector", "bm25"}:
            raise ValueError(f"Unsupported search mode: {mode}")

        filter_spec = FilterSpec.from_dict(filters)
        pool_limit = search_pool_limit(
            limit,
            filters_active=filter_spec.active,
            mode=mode,
        )
        query_embedding = None
        if mode in {"hybrid", "vector"}:
            query_embedding = await self._embedder.embed(query)
        fts_query = _fts5_query(query) if mode in {"hybrid", "bm25"} else ""

        def _work() -> dict[str, Any]:
            db = _open_db(self._db_path)
            try:
                vector_rows = []
                if query_embedding is not None:
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
                            pool_limit,
                            scope.user_id,
                            scope.namespace,
                        ),
                    ).fetchall()

                bm25_rows = []
                if fts_query:
                    bm25_rows = db.execute(
                        """
                        SELECT fts_memories.rowid, bm25(fts_memories) AS bm25_score
                        FROM fts_memories
                        JOIN memories ON memories.rowid = fts_memories.rowid
                        WHERE fts_memories MATCH ?
                          AND memories.user_id = ?
                          AND memories.namespace = ?
                        ORDER BY bm25_score
                        LIMIT ?
                        """,
                        (
                            fts_query,
                            scope.user_id,
                            scope.namespace,
                            pool_limit,
                        ),
                    ).fetchall()

                candidate_rowids = list(
                    dict.fromkeys(
                        [
                            int(row["rowid"])
                            for row in [*vector_rows, *bm25_rows]
                        ]
                    )
                )
                if not candidate_rowids:
                    return {
                        "rows": {},
                        "vector_ids": [],
                        "bm25_ids": [],
                        "distances": {},
                        "bm25_scores": {},
                    }

                placeholders = ",".join("?" for _ in candidate_rowids)
                subject_clause = ""
                parameters: list[Any] = list(candidate_rowids)
                if filter_spec.subject is not None:
                    subject_clause = "AND memories.subject = ? COLLATE NOCASE"
                    parameters.append(filter_spec.subject)
                rows = db.execute(
                    f"""
                    SELECT memories.rowid, {_memory_select_columns("memories")}
                    FROM memories
                    WHERE memories.rowid IN ({placeholders})
                      {subject_clause}
                    """,
                    tuple(parameters),
                ).fetchall()

                rows_by_id = {
                    str(int(row["rowid"])): dict(row)
                    for row in rows
                }
                distances = {
                    str(int(row["rowid"])): float(row["distance"])
                    for row in vector_rows
                }
                bm25_scores = {
                    str(int(row["rowid"])): float(row["bm25_score"])
                    for row in bm25_rows
                }
                return {
                    "rows": rows_by_id,
                    "vector_ids": [
                        str(int(row["rowid"]))
                        for row in vector_rows
                        if str(int(row["rowid"])) in rows_by_id
                    ],
                    "bm25_ids": [
                        str(int(row["rowid"]))
                        for row in bm25_rows
                        if str(int(row["rowid"])) in rows_by_id
                    ],
                    "distances": distances,
                    "bm25_scores": bm25_scores,
                }
            finally:
                db.close()

        candidates = await asyncio.to_thread(_work)
        rows_by_id = candidates["rows"]
        vector_ids = candidates["vector_ids"]
        bm25_ids = candidates["bm25_ids"]

        if mode == "vector":
            ranked = [
                ScoredMemory(
                    memory=_row_to_memory(rows_by_id[rowid]),
                    score=1.0 - candidates["distances"][rowid],
                )
                for rowid in vector_ids
            ]
        elif mode == "bm25":
            ranked = [
                ScoredMemory(
                    memory=_row_to_memory(rows_by_id[rowid]),
                    score=-candidates["bm25_scores"][rowid],
                )
                for rowid in bm25_ids
            ]
        else:
            fused_scores = reciprocal_rank_fusion([vector_ids, bm25_ids])
            fused_ids = sorted(
                fused_scores,
                key=lambda rowid: fused_scores[rowid],
                reverse=True,
            )
            ranked = [
                ScoredMemory(
                    memory=_row_to_memory(rows_by_id[rowid]),
                    score=fused_scores[rowid],
                )
                for rowid in fused_ids
            ]

        return ranked[:limit]

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
                    f"""
                    SELECT {_memory_select_columns("memories")}
                    FROM memories
                    WHERE memories.id = ?
                      AND memories.user_id = ?
                      AND memories.namespace = ?
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
                db.execute("BEGIN")
                row = db.execute(
                    """
                    SELECT rowid, content, subject
                    FROM memories
                    WHERE id = ? AND user_id = ? AND namespace = ?
                    """,
                    (id, scope.user_id, scope.namespace),
                ).fetchone()
                if row is None:
                    db.execute("ROLLBACK")
                    return False

                rowid = int(row["rowid"])
                _delete_fts_memory(
                    db,
                    rowid,
                    row["content"],
                    row["subject"],
                )
                db.execute("DELETE FROM vec_memories WHERE rowid = ?", (rowid,))
                db.execute("DELETE FROM memories WHERE rowid = ?", (rowid,))
                db.execute("COMMIT")
                return True
            except Exception:
                try:
                    db.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                db.close()

        return await asyncio.to_thread(_work)

    async def update(
        self,
        id: str,
        content: str,
        scope: Scope,
        *,
        internal_turn_ids: list[str] | None = None,
        subject: str | None = None,
        observed_at: str | None = None,
        source_session_id: str | None = None,
    ) -> Memory | None:
        existing = await self.get(id, scope)
        if existing is None:
            return None

        embedding = await self._embedder.embed(content)
        content_hash = _content_hash(content)
        updated_at = _utc_now()

        def _work() -> dict[str, Any] | None:
            db = _open_db(self._db_path)
            try:
                db.execute("BEGIN")
                row = db.execute(
                    """
                    SELECT rowid, content, subject
                    FROM memories
                    WHERE id = ? AND user_id = ? AND namespace = ?
                    """,
                    (id, scope.user_id, scope.namespace),
                ).fetchone()
                if row is None:
                    db.execute("ROLLBACK")
                    return None

                rowid = int(row["rowid"])
                next_subject = subject if subject is not None else row["subject"]
                _delete_fts_memory(
                    db,
                    rowid,
                    row["content"],
                    row["subject"],
                )
                db.execute(
                    """
                    UPDATE memories
                    SET content = ?,
                        subject = COALESCE(?, subject),
                        source_session_id = COALESCE(?, source_session_id),
                        observed_at = COALESCE(?, observed_at),
                        content_hash = ?,
                        updated_at = ?
                    WHERE rowid = ?
                    """,
                    (
                        content,
                        subject,
                        source_session_id,
                        observed_at,
                        content_hash,
                        updated_at,
                        rowid,
                    ),
                )
                _insert_fts_memory(db, rowid, content, next_subject)
                db.execute(
                    """
                    UPDATE vec_memories
                    SET embedding = ?
                    WHERE rowid = ?
                    """,
                    (sqlite_vec.serialize_float32(embedding), rowid),
                )
                if internal_turn_ids is not None:
                    db.execute(
                        "DELETE FROM memory_sources WHERE memory_id = ?",
                        (id,),
                    )
                    for turn_id in dict.fromkeys(internal_turn_ids):
                        db.execute(
                            """
                            INSERT INTO memory_sources(memory_id, turn_id)
                            VALUES (?, ?)
                            """,
                            (id, turn_id),
                        )
                updated = db.execute(
                    f"""
                    SELECT {_memory_select_columns("memories")}
                    FROM memories
                    WHERE memories.rowid = ?
                    """,
                    (rowid,),
                ).fetchone()
                db.execute("COMMIT")
                return dict(updated) if updated is not None else None
            except Exception:
                try:
                    db.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                db.close()

        row = await asyncio.to_thread(_work)
        return _row_to_memory(row) if row is not None else None
