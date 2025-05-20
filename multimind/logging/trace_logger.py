"""
Trace logging functionality for tracking execution flow.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Lis
from pathlib import Path

class TraceLogger:
    """Logs execution traces for debugging and analysis."""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_level: int = logging.INFO
    ):
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up file handler
        self.logger = logging.getLogger("multimind.trace")
        self.logger.setLevel(log_level)

        log_file = self.log_dir / f"trace_{datetime.now().strftime('%Y%m%d')}.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)

        # In-memory trace storage
        self.traces: List[Dict[str, Any]] = []

    def start_trace(
        self,
        trace_id: str,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Start a new trace."""
        trace = {
            "trace_id": trace_id,
            "operation": operation,
            "start_time": datetime.now().isoformat(),
            "metadata": metadata or {},
            "events": []
        }
        self.traces.append(trace)
        self.logger.info(f"Started trace {trace_id} for {operation}")

    def add_event(
        self,
        trace_id: str,
        event_type: str,
        data: Dict[str, Any],
        level: str = "info"
    ) -> None:
        """Add an event to a trace."""
        trace = self._get_trace(trace_id)
        if not trace:
            raise ValueError(f"Trace not found: {trace_id}")

        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data,
            "level": level
        }
        trace["events"].append(event)

        # Log even
        log_msg = f"Trace {trace_id} - {event_type}: {json.dumps(data)}"
        if level == "error":
            self.logger.error(log_msg)
        elif level == "warning":
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)

    def end_trace(
        self,
        trace_id: str,
        status: str = "success",
        result: Optional[Dict[str, Any]] = None
    ) -> None:
        """End a trace."""
        trace = self._get_trace(trace_id)
        if not trace:
            raise ValueError(f"Trace not found: {trace_id}")

        trace["end_time"] = datetime.now().isoformat()
        trace["status"] = status
        trace["result"] = resul

        # Save trace to file
        trace_file = self.log_dir / f"trace_{trace_id}.json"
        with open(trace_file, 'w') as f:
            json.dump(trace, f, indent=2)

        self.logger.info(
            f"Ended trace {trace_id} with status {status}"
        )

    def _get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get trace by ID."""
        for trace in self.traces:
            if trace["trace_id"] == trace_id:
                return trace
        return None

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get trace by ID from file."""
        trace_file = self.log_dir / f"trace_{trace_id}.json"
        if not trace_file.exists():
            return None

        with open(trace_file, 'r') as f:
            return json.load(f)

    def list_traces(
        self,
        operation: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List traces with optional filtering."""
        traces = []
        for trace_file in self.log_dir.glob("trace_*.json"):
            with open(trace_file, 'r') as f:
                trace = json.load(f)
                if operation and trace["operation"] != operation:
                    continue
                if status and trace.get("status") != status:
                    continue
                traces.append(trace)
        return traces