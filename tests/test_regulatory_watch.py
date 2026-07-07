"""Tests for RegulatoryWatcher — network mocked, no live HTTP calls."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from multimind.compliance.regulatory_watch import ChangeEvent, RegulatoryWatcher

SOURCES = [{"name": "gdpr", "url": "https://example.com/gdpr"}]


def make_response(text: str) -> httpx.Response:
    return httpx.Response(200, text=text, request=httpx.Request("GET", "https://example.com/gdpr"))


def make_mock_client(*texts: str) -> AsyncMock:
    client = AsyncMock()
    client.get.side_effect = [make_response(t) for t in texts]
    return client


@pytest.mark.asyncio
async def test_first_check_reports_no_change(tmp_path):
    client = make_mock_client("Article 1: initial text.")
    watcher = RegulatoryWatcher(SOURCES, state_path=str(tmp_path / "state.json"), client=client)

    events = await watcher.check()

    assert len(events) == 1
    assert isinstance(events[0], ChangeEvent)
    assert events[0].changed is False
    assert events[0].previous_hash is None
    assert events[0].diff == ""
    assert events[0].url == "https://example.com/gdpr"


@pytest.mark.asyncio
async def test_second_check_detects_change_with_diff(tmp_path):
    state_path = str(tmp_path / "state.json")
    client1 = make_mock_client("Article 1: initial text.")
    watcher1 = RegulatoryWatcher(SOURCES, state_path=state_path, client=client1)
    await watcher1.check()

    client2 = make_mock_client("Article 1: amended text.")
    watcher2 = RegulatoryWatcher(SOURCES, state_path=state_path, client=client2)
    events = await watcher2.check()

    assert events[0].changed is True
    assert events[0].previous_hash is not None
    assert events[0].previous_hash != events[0].current_hash
    assert "amended" in events[0].diff or "initial" in events[0].diff


@pytest.mark.asyncio
async def test_unchanged_content_no_diff(tmp_path):
    state_path = str(tmp_path / "state.json")
    client1 = make_mock_client("stable text")
    watcher1 = RegulatoryWatcher(SOURCES, state_path=state_path, client=client1)
    await watcher1.check()

    client2 = make_mock_client("stable text")
    watcher2 = RegulatoryWatcher(SOURCES, state_path=state_path, client=client2)
    events = await watcher2.check()

    assert events[0].changed is False
    assert events[0].diff == ""


@pytest.mark.asyncio
async def test_state_persisted_across_instances(tmp_path):
    state_path = str(tmp_path / "state.json")
    client1 = make_mock_client("v1")
    watcher1 = RegulatoryWatcher(SOURCES, state_path=state_path, client=client1)
    await watcher1.check()

    assert (tmp_path / "state.json").exists()

    watcher2 = RegulatoryWatcher(SOURCES, state_path=state_path, client=make_mock_client("v1"))
    assert "gdpr" in watcher2.state


@pytest.mark.asyncio
async def test_diff_truncated_to_diff_lines(tmp_path):
    state_path = str(tmp_path / "state.json")
    old_text = "\n".join(f"line {i}" for i in range(100))
    new_text = "\n".join(f"line {i} changed" for i in range(100))

    watcher1 = RegulatoryWatcher(
        SOURCES, state_path=state_path, client=make_mock_client(old_text), diff_lines=5
    )
    await watcher1.check()

    watcher2 = RegulatoryWatcher(
        SOURCES, state_path=state_path, client=make_mock_client(new_text), diff_lines=5
    )
    events = await watcher2.check()

    assert len(events[0].diff.splitlines()) <= 5


def test_check_offline_reads_local_files(tmp_path):
    state_path = str(tmp_path / "state.json")
    src_file = tmp_path / "gdpr.html"
    src_file.write_text("original content")

    watcher = RegulatoryWatcher(SOURCES, state_path=state_path)
    events = watcher.check_offline({"gdpr": str(src_file)})

    assert events[0].changed is False

    src_file.write_text("changed content")
    watcher2 = RegulatoryWatcher(SOURCES, state_path=state_path)
    events2 = watcher2.check_offline({"gdpr": str(src_file)})

    assert events2[0].changed is True
    assert events2[0].diff != ""


def test_check_offline_skips_sources_without_path(tmp_path):
    watcher = RegulatoryWatcher(SOURCES, state_path=str(tmp_path / "state.json"))

    events = watcher.check_offline({})

    assert events == []


def test_invalid_source_missing_url_rejected(tmp_path):
    with pytest.raises(ValueError, match="name.*url"):
        RegulatoryWatcher([{"name": "x"}], state_path=str(tmp_path / "state.json"))


@pytest.mark.asyncio
async def test_multiple_sources_independent_state(tmp_path):
    sources = [
        {"name": "gdpr", "url": "https://example.com/gdpr"},
        {"name": "hipaa", "url": "https://example.com/hipaa"},
    ]
    client = AsyncMock()
    client.get.side_effect = [
        httpx.Response(
            200, text="gdpr text", request=httpx.Request("GET", "https://example.com/gdpr")
        ),
        httpx.Response(
            200, text="hipaa text", request=httpx.Request("GET", "https://example.com/hipaa")
        ),
    ]
    watcher = RegulatoryWatcher(sources, state_path=str(tmp_path / "state.json"), client=client)

    events = await watcher.check()

    assert {e.name for e in events} == {"gdpr", "hipaa"}
    assert client.get.call_count == 2


@pytest.mark.asyncio
async def test_http_error_propagates(tmp_path):
    client = AsyncMock()
    error_response = httpx.Response(
        404, text="not found", request=httpx.Request("GET", "https://example.com/gdpr")
    )
    client.get.return_value = error_response
    watcher = RegulatoryWatcher(SOURCES, state_path=str(tmp_path / "state.json"), client=client)

    with pytest.raises(httpx.HTTPStatusError):
        await watcher.check()
