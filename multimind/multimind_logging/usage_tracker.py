"""
Usage tracking functionality for monitoring model usage and costs.
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

class UsageTracker:
    """Tracks model usage and associated costs."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        if db_path == ":memory:":
            self.conn = sqlite3.connect(db_path)
        else:
            self.conn = sqlite3.connect(str(self.db_path))
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the database with required tables."""
        print("Initializing database and creating tables if they do not exist...")
        cursor = self.conn.cursor()

        # Create costs table if it does not exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS costs (
                model TEXT PRIMARY KEY,
                input_cost_per_token REAL NOT NULL,
                output_cost_per_token REAL NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)
        print("Costs table creation attempted.")

        # Create usage table if it does not exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                operation TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL,
                metadata TEXT
            )
        """)
        print("Usage table creation attempted.")

        self.conn.commit()

    def track_usage(
        self,
        model: str,
        operation: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track model usage."""
        # Get costs for model
        input_cost_per_token, output_cost_per_token = self._get_model_costs(model)

        # Calculate cost
        cost = 0
        if input_tokens is not None:
            cost += input_tokens * input_cost_per_token
        if output_tokens is not None:
            cost += output_tokens * output_cost_per_token

        # Store usage
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO usage (
                timestamp, model, operation, input_tokens,
                output_tokens, cost, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            model,
            operation,
            input_tokens,
            output_tokens,
            cost,
            json.dumps(metadata) if metadata else None
        ))

        self.conn.commit()

    def set_model_costs(
        self,
        model: str,
        input_cost_per_token: float,
        output_cost_per_token: float
    ) -> None:
        """Set costs for a model."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO costs (
                model, input_cost_per_token, output_cost_per_token, last_updated
            ) VALUES (?, ?, ?, ?)
        """, (
            model,
            input_cost_per_token,
            output_cost_per_token,
            datetime.now().isoformat()
        ))
        self.conn.commit()

    def _get_model_costs(self, model: str) -> Tuple[float, float]:
        """Get costs for a model."""
        cursor = self.conn.cursor()

        cursor.execute(
            "SELECT input_cost_per_token, output_cost_per_token FROM costs WHERE model = ?",
            (model,)
        )
        result = cursor.fetchone()

        if result is None:
            # Default costs if not set
            return 0.0, 0.0

        return result

    def get_usage_summary(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get usage summary for a time period."""
        cursor = self.conn.cursor()

        # Build query
        query = "SELECT model, operation, COUNT(*) as count, "
        query += "SUM(input_tokens) as total_input_tokens, "
        query += "SUM(output_tokens) as total_output_tokens, "
        query += "SUM(cost) as total_cost "
        query += "FROM usage WHERE 1=1"

        params = []
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        if model:
            query += " AND model = ?"
            params.append(model)

        query += " GROUP BY model, operation"

        cursor.execute(query, params)
        results = cursor.fetchall()

        # Format results
        summary = {
            "total_cost": 0,
            "models": {}
        }

        for row in results:
            model, operation, count, input_tokens, output_tokens, cost = row

            if model not in summary["models"]:
                summary["models"][model] = {
                    "total_cost": 0,
                    "operations": {}
                }

            summary["models"][model]["operations"][operation] = {
                "count": count,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost
            }

            summary["models"][model]["total_cost"] += cost
            summary["total_cost"] += cost

        return summary

    def export_usage(
        self,
        file_path: str,
        format: str = "json",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> None:
        """Export usage data to file."""
        cursor = self.conn.cursor()

        # Build query
        query = "SELECT * FROM usage WHERE 1=1"
        params = []

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        cursor.execute(query, params)
        results = cursor.fetchall()

        # Get column names
        columns = [description[0] for description in cursor.description]

        # Format data
        data = []
        for row in results:
            item = dict(zip(columns, row))
            if item["metadata"]:
                item["metadata"] = json.loads(item["metadata"])
            data.append(item)

        # Export to file
        if format == "json":
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")