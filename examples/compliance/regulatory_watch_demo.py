"""Offline (--dry-run) regulatory change detection demo.

Simulates monitoring a regulatory source across two checks using local files
instead of live HTTP requests, so it runs with no network access or API keys.
`RegulatoryWatcher` only reports that content changed plus a raw diff — it
never fabricates a legal interpretation of what changed.
"""

import tempfile
from pathlib import Path

from multimind.compliance.regulatory_watch import RegulatoryWatcher


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        state_path = tmp_path / "regulatory_state.json"
        source_file = tmp_path / "gdpr_article_5.html"

        sources = [
            {"name": "gdpr_article_5", "url": "https://example.com/gdpr/article-5"},
        ]

        # First check: establishes the baseline, nothing to compare against yet.
        source_file.write_text(
            "Article 5: Personal data shall be processed lawfully, fairly and "
            "in a transparent manner."
        )
        watcher = RegulatoryWatcher(sources, state_path=str(state_path))
        events = watcher.check_offline({"gdpr_article_5": str(source_file)})
        print("First check:", events[0])

        # Second check: the source text changes.
        source_file.write_text(
            "Article 5: Personal data shall be processed lawfully, fairly, "
            "transparently, and with an explicit retention limit of 24 months."
        )
        watcher2 = RegulatoryWatcher(sources, state_path=str(state_path))
        events = watcher2.check_offline({"gdpr_article_5": str(source_file)})
        change = events[0]
        print("\nSecond check: changed =", change.changed)
        print("Diff (raw, no interpretation):")
        print(change.diff)


if __name__ == "__main__":
    main()
