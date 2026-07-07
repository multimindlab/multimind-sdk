"""Regulatory change detection via content hashing — no legal interpretation.

``RegulatoryWatcher`` fetches configured source pages (httpx, no JS
rendering), hashes their content, and reports a :class:`ChangeEvent` whenever
a source's hash differs from the last check. Events carry only the source
url, a ``changed`` flag, and a unified diff of the change — never a fabricated
summary of legal meaning or impact. Interpreting *what* a regulatory change
means requires a human or a downstream, clearly-labeled LLM step; this module
does neither.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class ChangeEvent:
    """Result of one source check. ``diff`` is a raw unified diff, no interpretation."""

    name: str
    url: str
    changed: bool
    diff: str
    checked_at: str
    previous_hash: Optional[str]
    current_hash: str


class RegulatoryWatcher:
    """Monitors regulatory source pages for content changes via hashing.

    Args:
        sources: list of ``{"name": ..., "url": ...}`` dicts identifying each
            page to monitor.
        state_path: JSON file persisting each source's last-seen hash/content.
        timeout: per-request timeout in seconds.
        diff_lines: max unified-diff lines kept per change event.
        client: optional pre-configured ``httpx.AsyncClient`` (e.g. for tests
            or custom auth); a fresh client is opened per check otherwise.
    """

    def __init__(
        self,
        sources: List[Dict[str, str]],
        state_path: str,
        timeout: float = 10.0,
        diff_lines: int = 40,
        client: Optional[httpx.AsyncClient] = None,
    ):
        for source in sources:
            if "name" not in source or "url" not in source:
                raise ValueError(f"Each source must have 'name' and 'url', got {source!r}")
        self.sources = sources
        self.state_path = Path(state_path)
        self.timeout = timeout
        self.diff_lines = diff_lines
        self._client = client
        self.state: Dict[str, Any] = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2))

    @staticmethod
    def _hash_content(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _diff_summary(self, old_content: str, new_content: str) -> str:
        diff = difflib.unified_diff(old_content.splitlines(), new_content.splitlines(), lineterm="")
        return "\n".join(list(diff)[: self.diff_lines])

    def _record(self, name: str, url: str, content: str, now: str) -> ChangeEvent:
        current_hash = self._hash_content(content)
        previous = self.state.get(name)
        previous_hash = previous.get("hash") if previous else None
        changed = previous_hash is not None and previous_hash != current_hash
        diff = self._diff_summary(previous.get("content", ""), content) if changed else ""
        self.state[name] = {"hash": current_hash, "content": content, "checked_at": now}
        return ChangeEvent(
            name=name,
            url=url,
            changed=changed,
            diff=diff,
            checked_at=now,
            previous_hash=previous_hash,
            current_hash=current_hash,
        )

    async def _fetch(self, url: str) -> str:
        if self._client is not None:
            response = await self._client.get(url, timeout=self.timeout)
        else:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(url)
        response.raise_for_status()
        return response.text

    async def check(self) -> List[ChangeEvent]:
        """Fetch every source over the network and report changes since last check."""
        now = datetime.now(timezone.utc).isoformat()
        events = []
        for source in self.sources:
            content = await self._fetch(source["url"])
            events.append(self._record(source["name"], source["url"], content, now))
        self._save_state()
        return events

    def check_offline(self, paths: Dict[str, str]) -> List[ChangeEvent]:
        """Dry-run mode: read local files instead of fetching over the network.

        ``paths`` maps source name to a local file path standing in for that
        source's page content. Sources without an entry in ``paths`` are
        skipped.
        """
        now = datetime.now(timezone.utc).isoformat()
        events = []
        for source in self.sources:
            name = source["name"]
            if name not in paths:
                continue
            content = Path(paths[name]).read_text()
            events.append(self._record(name, source["url"], content, now))
        self._save_state()
        return events
