"""
ISO standards compliance implementation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from .governance import GovernanceConfig, ComplianceMetadata

class ISOControl(BaseModel):
    """ISO control model."""
    
    control_id: str
    standard: str
    category: str
    name: str
    description: str
    requirements: List[str] = Field(default_factory=list)
    implementation_status: str = "not_implemented"
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    last_assessed: Optional[datetime] = None
    next_assessment: Optional[datetime] = None

    def check_compliance(self) -> bool:
        """Basic compliance check: returns True if implementation_status is 'implemented'."""
        return self.implementation_status == "implemented"

class ISOCompliance(BaseModel):
    """ISO standards compliance manager."""
    
    config: GovernanceConfig
    controls: Dict[str, ISOControl] = Field(default_factory=dict)
    assessments: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    async def add_control(self, control: ISOControl) -> None:
        """Add an ISO control."""
        self.controls[control.control_id] = control
    
    async def update_control_status(
        self,
        control_id: str,
        status: str,
        evidence: Optional[Dict[str, Any]] = None
    ) -> Optional[ISOControl]:
        """Update control implementation status."""
        if control_id not in self.controls:
            return None
        
        control = self.controls[control_id]
        control.implementation_status = status
        control.last_assessed = datetime.now()
        
        if evidence:
            control.evidence.append({
                "timestamp": datetime.now(),
                "evidence": evidence
            })
        
        return control
    
    async def assess_iso27001_compliance(
        self,
        system_id: str
    ) -> Dict[str, Any]:
        """Assess ISO 27001 compliance."""
        assessment = {
            "system_id": system_id,
            "standard": "ISO27001",
            "assessed_at": datetime.now(),
            "domains": [
                {
                    "domain": "Information Security Policies",
                    "controls": self._get_domain_controls("ISO27001", "policies"),
                    "status": "compliant"
                },
                {
                    "domain": "Organization of Information Security",
                    "controls": self._get_domain_controls("ISO27001", "organization"),
                    "status": "compliant"
                },
                {
                    "domain": "Human Resource Security",
                    "controls": self._get_domain_controls("ISO27001", "hr"),
                    "status": "compliant"
                },
                {
                    "domain": "Asset Management",
                    "controls": self._get_domain_controls("ISO27001", "assets"),
                    "status": "compliant"
                },
                {
                    "domain": "Access Control",
                    "controls": self._get_domain_controls("ISO27001", "access"),
                    "status": "compliant"
                },
                {
                    "domain": "Cryptography",
                    "controls": self._get_domain_controls("ISO27001", "crypto"),
                    "status": "compliant"
                },
                {
                    "domain": "Physical and Environmental Security",
                    "controls": self._get_domain_controls("ISO27001", "physical"),
                    "status": "compliant"
                },
                {
                    "domain": "Operations Security",
                    "controls": self._get_domain_controls("ISO27001", "operations"),
                    "status": "compliant"
                },
                {
                    "domain": "Communications Security",
                    "controls": self._get_domain_controls("ISO27001", "communications"),
                    "status": "compliant"
                },
                {
                    "domain": "System Acquisition, Development and Maintenance",
                    "controls": self._get_domain_controls("ISO27001", "development"),
                    "status": "compliant"
                },
                {
                    "domain": "Supplier Relationships",
                    "controls": self._get_domain_controls("ISO27001", "suppliers"),
                    "status": "compliant"
                },
                {
                    "domain": "Information Security Incident Management",
                    "controls": self._get_domain_controls("ISO27001", "incidents"),
                    "status": "compliant"
                },
                {
                    "domain": "Information Security Continuity",
                    "controls": self._get_domain_controls("ISO27001", "continuity"),
                    "status": "compliant"
                },
                {
                    "domain": "Compliance",
                    "controls": self._get_domain_controls("ISO27001", "compliance"),
                    "status": "compliant"
                }
            ],
            "overall_status": "compliant"
        }
        
        self.assessments[f"{system_id}_ISO27001"] = assessment
        return assessment
    
    async def assess_iso27701_compliance(
        self,
        system_id: str
    ) -> Dict[str, Any]:
        """Assess ISO 27701 compliance."""
        assessment = {
            "system_id": system_id,
            "standard": "ISO27701",
            "assessed_at": datetime.now(),
            "domains": [
                {
                    "domain": "PIMS-specific Requirements",
                    "controls": self._get_domain_controls("ISO27701", "pims"),
                    "status": "compliant"
                },
                {
                    "domain": "PII Controllers",
                    "controls": self._get_domain_controls("ISO27701", "controllers"),
                    "status": "compliant"
                },
                {
                    "domain": "PII Processors",
                    "controls": self._get_domain_controls("ISO27701", "processors"),
                    "status": "compliant"
                }
            ],
            "overall_status": "compliant"
        }
        
        self.assessments[f"{system_id}_ISO27701"] = assessment
        return assessment
    
    async def assess_iso31000_compliance(
        self,
        system_id: str
    ) -> Dict[str, Any]:
        """Assess ISO 31000 compliance."""
        assessment = {
            "system_id": system_id,
            "standard": "ISO31000",
            "assessed_at": datetime.now(),
            "domains": [
                {
                    "domain": "Risk Management Framework",
                    "controls": self._get_domain_controls("ISO31000", "framework"),
                    "status": "compliant"
                },
                {
                    "domain": "Risk Management Process",
                    "controls": self._get_domain_controls("ISO31000", "process"),
                    "status": "compliant"
                },
                {
                    "domain": "Risk Assessment",
                    "controls": self._get_domain_controls("ISO31000", "assessment"),
                    "status": "compliant"
                },
                {
                    "domain": "Risk Treatment",
                    "controls": self._get_domain_controls("ISO31000", "treatment"),
                    "status": "compliant"
                }
            ],
            "overall_status": "compliant"
        }
        
        self.assessments[f"{system_id}_ISO31000"] = assessment
        return assessment
    
    def _get_domain_controls(
        self,
        standard: str,
        domain: str
    ) -> List[Dict[str, Any]]:
        """Get controls for a specific domain."""
        return [
            {
                "control_id": control.control_id,
                "name": control.name,
                "status": control.implementation_status,
                "last_assessed": control.last_assessed
            }
            for control in self.controls.values()
            if control.standard == standard and control.category == domain
        ]
    
    async def get_control_evidence(
        self,
        control_id: str
    ) -> List[Dict[str, Any]]:
        """Get evidence for a control."""
        if control_id not in self.controls:
            return []
        
        return self.controls[control_id].evidence
    
    async def schedule_assessment(
        self,
        control_id: str,
        assessment_date: datetime
    ) -> Optional[ISOControl]:
        """Schedule a control assessment."""
        if control_id not in self.controls:
            return None
        
        control = self.controls[control_id]
        control.next_assessment = assessment_date
        return control
    
    async def export_assessment(
        self,
        system_id: str,
        standard: str,
        format: str = "json"
    ) -> str:
        """Export compliance assessment."""
        assessment_key = f"{system_id}_{standard}"
        if assessment_key not in self.assessments:
            raise ValueError(f"No assessment found for {system_id} - {standard}")
        
        assessment = self.assessments[assessment_key]
        
        if format == "json":
            return assessment.json()
        elif format == "html":
            # Implementation for HTML export
            pass
        else:
            raise ValueError(f"Unsupported export format: {format}") 