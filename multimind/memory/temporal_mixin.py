from datetime import datetime
from typing import Any, Dict, List, Optional

class TemporalMemoryMixin:
    """
    Mixin to add timestamps to memory entries and provide timeline tracking/querying.
    Can be used with any memory class that stores entries as dicts or objects.
    """
    def add_with_timestamp(self, entry: Dict[str, Any], timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Add a timestamp to the entry (in-place) and return it.
        """
        entry['timestamp'] = timestamp or datetime.utcnow()
        return entry

    def get_timeline(self, entries: List[Dict[str, Any]], since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Return entries added/updated since a given time, sorted by timestamp.
        """
        timeline = [e for e in entries if 'timestamp' in e and (since is None or e['timestamp'] >= since)]
        return sorted(timeline, key=lambda x: x['timestamp']) 