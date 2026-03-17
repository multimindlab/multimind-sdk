import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

class TraceLogger:
    """
    Logs agent actions, inputs, outputs, and memory updates with timestamps and context.
    Can be plugged into agents, pipelines, or memory modules for developer-friendly tracing.
    """
    def __init__(self, name: str = 'TraceLogger', log_to_console: bool = True):
        self.name = name
        self.log_to_console = log_to_console
        self.logs: List[Dict[str, Any]] = []
        if log_to_console:
            self.logger = logging.getLogger(name)
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler()
                formatter = logging.Formatter('[%(asctime)s] %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = None

    def log(self, action: str, agent: Optional[str] = None, input_data: Any = None, output_data: Any = None, memory_update: Any = None, context: Optional[Dict[str, Any]] = None):
        """
        Log an action with optional agent, input, output, memory update, and context.
        """
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'agent': agent,
            'input': input_data,
            'output': output_data,
            'memory_update': memory_update,
            'context': context or {}
        }
        self.logs.append(entry)
        if self.logger:
            msg = f"[{action}] agent={agent} input={input_data} output={output_data} memory_update={memory_update} context={context}"
            self.logger.info(msg)

    def get_logs(self, filter_action: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all logs, optionally filtered by action type.
        """
        if filter_action:
            return [log for log in self.logs if log['action'] == filter_action]
        return self.logs

    def clear(self):
        """
        Clear all logs.
        """
        self.logs.clear() 