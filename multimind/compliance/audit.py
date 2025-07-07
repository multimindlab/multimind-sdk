"""
Compliance audit logging implementation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from .governance import GovernanceConfig
import json

class AuditEvent(BaseModel):
    """Audit event model."""
    
    event_id: str
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None
    system_id: Optional[str] = None
    data_id: Optional[str] = None
    action: str
    details: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ComplianceAuditLogger(BaseModel):
    """Compliance audit logger."""
    
    config: GovernanceConfig
    events: List[AuditEvent] = Field(default_factory=list)
    
    async def log_event(
        self,
        event_type: str,
        action: str,
        user_id: Optional[str] = None,
        system_id: Optional[str] = None,
        data_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log a compliance audit event."""
        event = AuditEvent(
            event_id=f"evt_{len(self.events) + 1}",
            event_type=event_type,
            user_id=user_id,
            system_id=system_id,
            data_id=data_id,
            action=action,
            details=details or {},
            metadata=metadata or {}
        )
        
        self.events.append(event)
        return event
    
    async def get_events(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        system_id: Optional[str] = None,
        data_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditEvent]:
        """Get filtered audit events."""
        filtered_events = self.events
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        if system_id:
            filtered_events = [e for e in filtered_events if e.system_id == system_id]
        if data_id:
            filtered_events = [e for e in filtered_events if e.data_id == data_id]
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        return filtered_events
    
    async def get_user_activity(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditEvent]:
        """Get user activity audit trail."""
        return await self.get_events(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time
        )
    
    async def get_system_activity(
        self,
        system_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditEvent]:
        """Get system activity audit trail."""
        return await self.get_events(
            system_id=system_id,
            start_time=start_time,
            end_time=end_time
        )
    
    async def get_data_access(
        self,
        data_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditEvent]:
        """Get data access audit trail."""
        return await self.get_events(
            data_id=data_id,
            start_time=start_time,
            end_time=end_time
        )
    
    async def cleanup_old_events(self) -> int:
        """Clean up events older than retention period."""
        retention_date = datetime.now() - timedelta(days=self.config.audit_log_retention_days)
        old_events = [e for e in self.events if e.timestamp < retention_date]
        self.events = [e for e in self.events if e.timestamp >= retention_date]
        return len(old_events)
    
    async def export_events(
        self,
        format: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> str:
        """Export audit events in specified format."""
        events = await self.get_events(start_time=start_time, end_time=end_time)
        
        if format == "json":
            return json.dumps([e.dict() for e in events], default=str)
        elif format == "csv":
            import csv
            import io
            if not events:
                return ""
            output = io.StringIO()
            fieldnames = list(events[0].dict().keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for e in events:
                row = e.dict()
                # Convert datetime to string for CSV
                for k, v in row.items():
                    if isinstance(v, datetime):
                        row[k] = v.isoformat()
                writer.writerow(row)
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def get_compliance_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate compliance report from audit events."""
        events = await self.get_events(start_time=start_time, end_time=end_time)
        
        report = {
            "generated_at": datetime.now(),
            "period": {
                "start": start_time,
                "end": end_time
            },
            "summary": {
                "total_events": len(events),
                "event_types": {},
                "user_activity": {},
                "system_activity": {},
                "data_access": {}
            }
        }
        
        # Aggregate statistics
        for event in events:
            # Event types
            report["summary"]["event_types"][event.event_type] = \
                report["summary"]["event_types"].get(event.event_type, 0) + 1
            
            # User activity
            if event.user_id:
                report["summary"]["user_activity"][event.user_id] = \
                    report["summary"]["user_activity"].get(event.user_id, 0) + 1
            
            # System activity
            if event.system_id:
                report["summary"]["system_activity"][event.system_id] = \
                    report["summary"]["system_activity"].get(event.system_id, 0) + 1
            
            # Data access
            if event.data_id:
                report["summary"]["data_access"][event.data_id] = \
                    report["summary"]["data_access"].get(event.data_id, 0) + 1
        
        return report 