"""
Full-text search must find identifiers containing path/version separators.

The tsvector trigger indexed content verbatim, and the Postgres parser keeps
slash/dot/dash/tilde sequences together as URL-like lexemes:
"ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU_MODEL" produced the single lexeme
"opus/sonnet/haiku_model". The query side (_build_tsquery) splits on every
non-alphanumeric, so an exact-identifier query ANDed plain words that no
chunk could ever match — the full-text arm returned zero rows and a chunk
containing the identifier verbatim was invisible to keyword search.

The trigger must normalise the separators the query tokenizer splits on so
both sides agree about word boundaries, and the migration must re-lex
existing rows the same way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_vault.mcp import server as mcp_server
from memory_vault.models.db import execute_query, fetch_one
from memory_vault.services.search import _build_tsquery

pytestmark = pytest.mark.asyncio

MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "memory_vault"
    / "migrations"
    / "010_fts_separator_normalization.sql"
)


async def _store(text: str) -> str:
    import json

    return json.loads(await mcp_server.remember(text=text))["chunk_id"]


async def _lexemes() -> set[str]:
    row = await fetch_one("SELECT tsvector_to_array(content_tsv) AS lex FROM chunks LIMIT 1")
    return set(row["lex"])


async def _fts_matches(tsquery: str) -> int:
    row = await fetch_one(
        "SELECT count(*) AS n FROM chunks WHERE content_tsv @@ to_tsquery('english', %s)",
        (tsquery,),
    )
    return row["n"]


async def test_slash_compound_splits_into_word_lexemes():
    await _store("Pinned models: ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU_MODEL.")

    lex = await _lexemes()
    assert "opus" in lex
    assert "sonnet" in lex
    assert "haiku" in lex
    assert not any("/" in word for word in lex), f"slash survived in lexemes: {lex}"


async def test_exact_identifier_tsquery_finds_chunk():
    # The query side splits "ANTHROPIC_DEFAULT_OPUS_MODEL" into plain words;
    # before the fix the content side kept "opus/sonnet/haiku_model" whole,
    # so this AND-query matched zero rows even though the identifier appears
    # verbatim in the chunk.
    await _store("The failing pin was ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU_MODEL.")

    tsq = _build_tsquery("ANTHROPIC_DEFAULT_OPUS_MODEL")
    assert tsq is not None
    assert await _fts_matches(tsq) == 1


async def test_version_string_is_findable():
    # Letter-digit adjacency ("v1.18.28") is a separate, known limitation:
    # this covers the separator normalisation itself.
    await _store("Upgraded the vault stack to 1.18.28 today.")

    tsq = _build_tsquery("1.18.28")
    assert tsq is not None
    assert await _fts_matches(tsq) == 1


async def test_hyphenated_name_is_findable():
    await _store("The project lives in the memory-vault repository.")

    lex = await _lexemes()
    assert not any("-" in word for word in lex), f"hyphen survived in lexemes: {lex}"

    tsq = _build_tsquery("memory vault repository")
    assert tsq is not None
    assert await _fts_matches(tsq) == 1


async def test_tilde_alias_is_findable():
    await _store("Subagent pin: ~anthropic/claude-opus-latest via OpenRouter.")

    tsq = _build_tsquery("claude opus latest")
    assert tsq is not None
    assert await _fts_matches(tsq) == 1


async def test_migration_re_lexes_rows_indexed_by_old_trigger():
    await _store("Legacy row with openrouter/qwen/qwen3.8-2.4t-a95b inside.")

    # Simulate a row written before migration 010: the old trigger lexed the
    # content verbatim. Assigning content_tsv directly does not fire the
    # trigger (it only fires on INSERT or UPDATE OF content).
    await execute_query(
        "UPDATE chunks SET content_tsv = to_tsvector('english', COALESCE(content, ''))",
        commit=True,
    )
    assert any("/" in word for word in await _lexemes()), "setup should recreate old-style lexing"

    # Re-applying the migration fixes legacy rows.
    await execute_query(MIGRATION.read_text(), commit=True)

    lex = await _lexemes()
    assert not any("/" in word for word in lex), f"backfill left slash lexemes: {lex}"
    tsq = _build_tsquery("qwen3 8 2 4t a95b")
    assert tsq is not None
    assert await _fts_matches(tsq) == 1
