"""
Full-text search must find identifiers containing path/version separators.

The tsvector trigger indexed content verbatim, and the Postgres parser keeps
punctuation-bearing sequences together as URL/host/email-like lexemes:
"ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU_MODEL" produced the single lexeme
"opus/sonnet/haiku_model". The query side (_build_tsquery) splits on every
non-alphanumeric, so an exact-identifier query ANDed plain words that no
chunk could ever match — the full-text arm returned zero rows and a chunk
containing the identifier verbatim was invisible to keyword search.

The trigger must normalise the same way the query tokenizer splits so both
sides agree about word boundaries, and the migration must re-lex existing
rows with the identical expression.
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


async def _lexemes(chunk_id: str) -> set[str]:
    row = await fetch_one(
        "SELECT tsvector_to_array(content_tsv) AS lex FROM chunks WHERE id = %s",
        (chunk_id,),
    )
    return set(row["lex"])


async def _fts_matches(tsquery: str, chunk_id: str) -> bool:
    row = await fetch_one(
        "SELECT content_tsv @@ to_tsquery('english', %s) AS hit FROM chunks WHERE id = %s",
        (tsquery, chunk_id),
    )
    return row["hit"]


async def test_slash_compound_splits_into_word_lexemes():
    cid = await _store("Pinned models: ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU_MODEL.")

    lex = await _lexemes(cid)
    assert "opus" in lex
    assert "sonnet" in lex
    assert "haiku" in lex
    assert not any("/" in word for word in lex), f"slash survived in lexemes: {lex}"


async def test_split_identifier_query_finds_chunk():
    # The live failure this migration fixes: querying one identifier form
    # while the chunk stores another with the same words. The query side
    # ANDs plain words; before the fix the content side kept
    # "opus/sonnet/haiku_model" whole, so the AND matched zero rows.
    # Word-level semantics (not phrase): any chunk carrying all the words
    # matches — that is _build_tsquery's existing contract.
    cid = await _store("The failing pin was ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU_MODEL.")

    tsq = _build_tsquery("ANTHROPIC_DEFAULT_OPUS_MODEL")
    assert await _fts_matches(tsq, cid)


async def test_version_string_is_findable():
    # Letter-digit adjacency ("v1.18.28") is a separate, known limitation:
    # this covers the separator normalisation itself.
    cid = await _store("Upgraded the vault stack to 1.18.28 today.")

    tsq = _build_tsquery("1.18.28")
    assert await _fts_matches(tsq, cid)


async def test_hyphenated_compound_lexeme_removed():
    # Not a recall fix: the parser already emits hyphen-part lexemes
    # ("memory-vault" -> memori + memory-vault + vault), so recall worked
    # before. Normalisation removes the compound lexeme so the index is
    # consistent with the other separator classes.
    cid = await _store("The project lives in the memory-vault repository.")

    lex = await _lexemes(cid)
    assert not any("-" in word for word in lex), f"hyphen survived in lexemes: {lex}"

    tsq = _build_tsquery("memory vault repository")
    assert await _fts_matches(tsq, cid)


async def test_tilde_alias_is_findable():
    cid = await _store("Subagent pin: ~anthropic/claude-opus-latest via OpenRouter.")

    tsq = _build_tsquery("claude opus latest")
    assert await _fts_matches(tsq, cid)


async def test_email_and_host_forms_split():
    cid = await _store("Mail hmodes@example.com and see host db.internal for details.")

    # Stemmed forms: the english dictionary stems tsvector and tsquery alike.
    lex = await _lexemes(cid)
    assert "hmode" in lex
    assert "exampl" in lex
    assert "com" in lex
    assert not any("@" in word for word in lex), f"email survived in lexemes: {lex}"


async def test_unicode_words_survive_normalisation():
    ctype = await fetch_one("SELECT datctype FROM pg_database WHERE datname = current_database()")
    if "utf8" not in ctype["datctype"].lower():
        pytest.skip(f"[[:alnum:]] is ASCII-only under ctype {ctype['datctype']}")

    cid = await _store("Déployé le café service with a naïve caching layer.")

    # Accented words must stay whole (no ASCII mangling); the stemmer still
    # applies its usual rules ("naïve" -> "naïv").
    lex = await _lexemes(cid)
    assert "déployé" in lex
    assert "café" in lex
    assert "naïv" in lex


async def test_migration_re_lexes_rows_indexed_by_old_trigger():
    cid = await _store("Legacy row with openrouter/qwen/qwen3.8-2.4t-a95b inside.")

    # Simulate a row written before migration 010: the old trigger lexed the
    # content verbatim. Assigning content_tsv directly does not fire the
    # trigger (it only fires on INSERT or UPDATE OF content).
    await execute_query(
        "UPDATE chunks SET content_tsv = to_tsvector('english', COALESCE(content, '')) "
        "WHERE id = %s",
        (cid,),
        commit=True,
    )
    assert any("/" in word for word in await _lexemes(cid)), (
        "setup should recreate old-style lexing"
    )

    # Re-applying the migration fixes legacy rows.
    await execute_query(MIGRATION.read_text(), commit=True)

    lex = await _lexemes(cid)
    assert not any("/" in word for word in lex), f"backfill left slash lexemes: {lex}"
    tsq = _build_tsquery("qwen3.8-2.4t-a95b")
    assert await _fts_matches(tsq, cid)


async def test_backfill_expression_matches_trigger_expression():
    # The migration's backfill and the trigger must produce identical
    # tsvectors for the same content; if the two expressions drift, this
    # corpus (every supported separator class plus edge cases) catches it:
    # re-running the migration must not change a trigger-produced vector.
    corpus = (
        "Corpus: ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU_MODEL v1.18.28 memory-vault "
        "~anthropic/claude-opus-latest hmodes@example.com a+b c:d #tag 192.168.0.1 "
        "café naïve //--.. trailing-"
    )
    cid = await _store(corpus)
    before = await fetch_one(
        "SELECT content_tsv::text AS tsv, xmin FROM chunks WHERE id = %s", (cid,)
    )

    await execute_query(MIGRATION.read_text(), commit=True)

    after = await fetch_one(
        "SELECT content_tsv::text AS tsv, xmin FROM chunks WHERE id = %s", (cid,)
    )
    assert before["tsv"] == after["tsv"], (
        "backfill expression disagrees with the trigger; a re-run of the "
        "migration must be a no-op on trigger-produced rows"
    )
    # xmin changes on any row rewrite: proving equality is not enough, the
    # IS DISTINCT FROM guard must actually SKIP the row.
    assert before["xmin"] == after["xmin"], "unchanged row was rewritten anyway"


async def test_update_path_re_lexes_content():
    # The trigger fires on UPDATE OF content too, not only INSERT.
    cid = await _store("Original wording with model/pin separators.")

    await execute_query(
        "UPDATE chunks SET content = %s WHERE id = %s",
        ("Replaced wording about openrouter/qwen/qwen3.8-2.4t-a95b.", cid),
        commit=True,
    )

    tsq = _build_tsquery("openrouter qwen qwen3")
    assert await _fts_matches(tsq, cid)
    assert "separators" not in await _lexemes(cid)


async def test_recall_finds_identifier_end_to_end():
    # The user-visible symptom, through the real search path: before the fix
    # the FTS arm contributed nothing for identifier queries and the vector
    # arm alone missed them on a small corpus.
    import json

    stored = json.loads(
        await mcp_server.remember(
            text="Pinned ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU_MODEL in the shell rc."
        )
    )

    res = json.loads(await mcp_server.recall(query="ANTHROPIC_DEFAULT_OPUS_MODEL", limit=10))
    assert stored["chunk_id"] in [r["chunk_id"] for r in res["results"]]
