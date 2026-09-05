"""
recall_exact — literal substring search over stored memories.

recall's hybrid ranking can miss exact identifiers even when the words are
present; recall_exact is the ground-truth complement: a case-insensitive
substring match with no embeddings. These tests pin its semantics —
literal matching with LIKE wildcards escaped, forgotten chunks excluded,
and space filters with the same unknown-space semantics as recall.
"""

from __future__ import annotations

import json

import pytest

from memory_vault.mcp import server as mcp_server

pytestmark = pytest.mark.asyncio


async def _store(text: str, space: str = "default") -> str:
    return json.loads(await mcp_server.remember(text=text, space=space))["chunk_id"]


async def _ids(response: str) -> list[str]:
    return [r["chunk_id"] for r in json.loads(response)["results"]]


async def test_finds_literal_substring_case_insensitively():
    cid = await _store("The dhcpd dispatcher lives at /etc/NetworkManager/dispatcher.d/.")

    ids = await _ids(await mcp_server.recall_exact(query="DHCPD DISPATCHER"))
    assert cid in ids


async def test_underscore_is_literal_not_wildcard():
    kept = await _store("The pin is FOO_BAR in the rc file.")
    await _store("The pin is FOOXBAR in the rc file.")

    ids = await _ids(await mcp_server.recall_exact(query="FOO_BAR"))
    assert kept in ids
    assert len(ids) == 1, "underscore must not act as a single-character wildcard"


async def test_percent_is_literal_not_wildcard():
    kept = await _store("CPU steady at 100% load on operator.")
    await _store("The qwen-100b model scored well in the eval.")

    ids = await _ids(await mcp_server.recall_exact(query="100%"))
    assert kept in ids
    assert len(ids) == 1, "percent must not act as a wildcard"


async def test_backslash_is_literal():
    kept = await _store(r"Windows path C:\temp\ kept verbatim.")
    await _store("Windows path C temp kept verbatim.")

    ids = await _ids(await mcp_server.recall_exact(query=r"C:\temp"))
    assert kept in ids
    assert len(ids) == 1

    trailing = await _ids(await mcp_server.recall_exact(query="temp\\"))
    assert trailing == [kept]


async def test_backslash_wildcard_combinations_are_literal():
    # Negative control: if escaping were broken, an unescaped \% / \_ would
    # degrade to literal % / _ and match this chunk too.
    kept = await _store(r"Literal \% and \_ sequences stay intact here.")
    await _store("Plain percent and underscore: % and _ without backslashes.")

    ids = await _ids(await mcp_server.recall_exact(query=r"\% and \_"))
    assert kept in ids
    assert len(ids) == 1, "backslash-wildcard combos must match literally"


async def test_forgotten_chunks_excluded():
    cid = await _store("Transient note about the zfs scrub schedule.")
    assert cid in await _ids(await mcp_server.recall_exact(query="zfs scrub"))

    await mcp_server.forget(cid)
    assert cid not in await _ids(await mcp_server.recall_exact(query="zfs scrub"))


async def test_space_filter_and_unknown_space():
    from memory_vault.services.spaces import ensure_space

    await ensure_space("projects")
    cid = await _store("Deploy key rotated quarterly.", space="projects")

    ids = await _ids(await mcp_server.recall_exact(query="deploy key rotated", spaces=["projects"]))
    assert cid in ids

    assert await _ids(await mcp_server.recall_exact(query="deploy key rotated")) == [cid]
    assert (
        await _ids(await mcp_server.recall_exact(query="deploy key rotated", spaces=["default"]))
        == []
    )
    # Unknown space: zero rows, never a silent widening to every space.
    # Unknown space: zero rows, never a silent widening to every space, and
    # the response shape stays identical to a normal search.
    res = json.loads(
        await mcp_server.recall_exact(query="deploy key rotated", spaces=["no-such-space"])
    )
    assert res["status"] == "ok"
    assert res["results"] == []
    assert res["total_results"] == 0
    assert "query_time_ms" in res


async def test_limit_caps_results():
    for i in range(5):
        await _store(f"Benchmark run number {i} of the nightly suite.")

    res = json.loads(await mcp_server.recall_exact(query="nightly suite", limit=2))
    assert len(res["results"]) == 2
    assert res["total_results"] == 2


async def test_empty_query_rejected():
    res = json.loads(await mcp_server.recall_exact(query="   "))
    assert res["status"] == "error"
