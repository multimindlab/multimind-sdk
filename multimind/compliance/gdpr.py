"""
GDPR compliance implementation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from .governance import GovernanceConfig, ComplianceMetadata, DataCategory

class GDPRCompliance(BaseModel):
    """GDPR compliance manager."""
    
    config: GovernanceConfig
    data_registry: Dict[str, ComplianceMetadata] = Field(default_factory=dict)
    
    async def process_data(
        self,
        data_id: str,
        content: Any,
        metadata: Dict[str, Any]
    ) -> ComplianceMetadata:
        """Process data according to GDPR requirements."""
        # Create compliance metadata
        compliance_metadata = ComplianceMetadata(
            data_category=metadata.get("data_category", DataCategory.PERSONAL),
            risk_level=metadata.get("risk_level"),
            regulation_tags=[Regulation.GDPR],
            lawful_basis=metadata.get("lawful_basis"),
            consent_granted=metadata.get("consent_granted", False),
            data_subject_id=metadata.get("data_subject_id"),
            expires_at=datetime.now() + timedelta(days=self.config.data_retention_days)
        )
        
        # Store metadata
        self.data_registry[data_id] = compliance_metadata
        
        return compliance_metadata
    
    async def handle_dsar(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle Data Subject Access Request."""
        # Find all data for subject
        subject_data = {
            data_id: metadata
            for data_id, metadata in self.data_registry.items()
            if metadata.data_subject_id == data_subject_id
        }
        
        return {
            "data_subject_id": data_subject_id,
            "requested_at": datetime.now(),
            "data_items": subject_data
        }
    
    async def handle_erasure(self, data_subject_id: str) -> bool:
        """Handle data erasure request."""
        # Find and remove all data for subject
        data_to_remove = [
            data_id
            for data_id, metadata in self.data_registry.items()
            if metadata.data_subject_id == data_subject_id
        ]
        
        for data_id in data_to_remove:
            del self.data_registry[data_id]
        
        return len(data_to_remove) > 0
    
    async def check_retention(self) -> List[str]:
        """Check for data that needs to be deleted due to retention policy."""
        now = datetime.now()
        expired_data = [
            data_id
            for data_id, metadata in self.data_registry.items()
            if metadata.expires_at and metadata.expires_at < now
        ]
        
        return expired_data
    
    async def validate_lawful_basis(
        self,
        data_category: DataCategory,
        lawful_basis: str,
        consent_granted: bool
    ) -> bool:
        """Validate if the lawful basis is appropriate for the data category."""
        if data_category in [DataCategory.PERSONAL, DataCategory.SENSITIVE]:
            if not lawful_basis or not consent_granted:
                return False
        
        return True
    
    async def get_processing_activities(self) -> List[Dict[str, Any]]:
        """Get record of processing activities."""
        return [
            {
                "data_id": data_id,
                "metadata": metadata.dict(),
                "last_accessed": metadata.last_accessed,
                "access_count": metadata.access_count
            }
            for data_id, metadata in self.data_registry.items()
        ]
    
    async def update_metadata(
        self,
        data_id: str,
        updates: Dict[str, Any]
    ) -> Optional[ComplianceMetadata]:
        """Update compliance metadata for data."""
        if data_id not in self.data_registry:
            return None
        
        metadata = self.data_registry[data_id]
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        metadata.version += 1
        return metadata 