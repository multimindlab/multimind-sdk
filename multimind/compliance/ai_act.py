"""
EU AI Act compliance implementation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from .governance import GovernanceConfig, ComplianceMetadata, RiskLevel

class AIActCompliance(BaseModel):
    """EU AI Act compliance manager."""
    
    config: GovernanceConfig
    risk_assessments: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    technical_docs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    async def assess_risk(
        self,
        system_id: str,
        system_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform risk assessment for AI system."""
        # Determine risk level
        risk_level = self._determine_risk_level(system_metadata)
        
        # Create risk assessment
        assessment = {
            "system_id": system_id,
            "risk_level": risk_level,
            "assessment_date": datetime.now(),
            "metadata": system_metadata,
            "findings": [],
            "recommendations": []
        }
        
        # Add findings based on risk level
        if risk_level == RiskLevel.HIGH:
            assessment["findings"].extend([
                "System requires conformity assessment",
                "Technical documentation required",
                "Quality management system required",
                "Post-market monitoring required"
            ])
            assessment["recommendations"].extend([
                "Implement risk management system",
                "Maintain technical documentation",
                "Enable human oversight",
                "Implement logging and monitoring"
            ])
        
        # Store assessment
        self.risk_assessments[system_id] = assessment
        
        return assessment
    
    async def generate_technical_docs(
        self,
        system_id: str,
        system_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate technical documentation for AI system."""
        # Create technical documentation
        docs = {
            "system_id": system_id,
            "generated_at": datetime.now(),
            "system_details": system_details,
            "sections": {
                "system_description": self._generate_system_description(system_details),
                "risk_management": self._generate_risk_management(system_id),
                "data_governance": self._generate_data_governance(system_details),
                "technical_specifications": self._generate_tech_specs(system_details),
                "testing_results": self._generate_testing_results(system_details),
                "post_market_monitoring": self._generate_monitoring_plan(system_id)
            }
        }
        
        # Store documentation
        self.technical_docs[system_id] = docs
        
        return docs
    
    async def validate_compliance(
        self,
        system_id: str
    ) -> Dict[str, Any]:
        """Validate system compliance with AI Act requirements."""
        if system_id not in self.risk_assessments:
            raise ValueError(f"No risk assessment found for system {system_id}")
        
        assessment = self.risk_assessments[system_id]
        validation = {
            "system_id": system_id,
            "validated_at": datetime.now(),
            "risk_level": assessment["risk_level"],
            "requirements": [],
            "status": "compliant"
        }
        
        # Check requirements based on risk level
        if assessment["risk_level"] == RiskLevel.HIGH:
            requirements = [
                "technical_documentation",
                "risk_management_system",
                "quality_management_system",
                "post_market_monitoring",
                "human_oversight",
                "logging_and_monitoring"
            ]
            
            for req in requirements:
                status = self._check_requirement(system_id, req)
                validation["requirements"].append({
                    "requirement": req,
                    "status": status
                })
                if status != "compliant":
                    validation["status"] = "non_compliant"
        
        return validation
    
    def _determine_risk_level(self, metadata: Dict[str, Any]) -> RiskLevel:
        """Determine risk level based on system metadata."""
        # Check for unacceptable risk
        if metadata.get("is_social_scoring") or metadata.get("is_biometric_id"):
            return RiskLevel.UNACCEPTABLE
        
        # Check for high risk
        if metadata.get("is_medical_device") or metadata.get("is_critical_infrastructure"):
            return RiskLevel.HIGH
        
        # Check for limited risk
        if metadata.get("is_chatbot") or metadata.get("is_emotion_recognition"):
            return RiskLevel.LIMITED
        
        return RiskLevel.MINIMAL
    
    def _generate_system_description(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate system description section."""
        return {
            "purpose": details.get("purpose", "Not specified"),
            "capabilities": details.get("capabilities", []),
            "limitations": details.get("limitations", []),
            "intended_users": details.get("intended_users", []),
            "deployment_context": details.get("deployment_context", {})
        }
    
    def _generate_risk_management(self, system_id: str) -> Dict[str, Any]:
        """Generate risk management section."""
        assessment = self.risk_assessments.get(system_id, {})
        return {
            "risk_level": assessment.get("risk_level"),
            "identified_risks": assessment.get("findings", []),
            "mitigation_strategies": assessment.get("recommendations", []),
            "monitoring_measures": []
        }
    
    def _generate_data_governance(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data governance section."""
        return {
            "data_sources": details.get("data_sources", []),
            "data_processing": details.get("data_processing", {}),
            "data_quality": details.get("data_quality", {}),
            "data_protection": details.get("data_protection", {})
        }
    
    def _generate_tech_specs(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical specifications section."""
        return {
            "architecture": details.get("architecture", {}),
            "algorithms": details.get("algorithms", []),
            "performance_metrics": details.get("performance_metrics", {}),
            "system_requirements": details.get("system_requirements", {})
        }
    
    def _generate_testing_results(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Generate testing results section."""
        return {
            "test_cases": details.get("test_cases", []),
            "performance_results": details.get("performance_results", {}),
            "validation_results": details.get("validation_results", {}),
            "certification_status": details.get("certification_status", {})
        }
    
    def _generate_monitoring_plan(self, system_id: str) -> Dict[str, Any]:
        """Generate post-market monitoring plan."""
        return {
            "monitoring_metrics": [],
            "incident_reporting": {},
            "update_procedures": {},
            "user_feedback": {}
        }
    
    def _check_requirement(self, system_id: str, requirement: str) -> str:
        """Check if a specific requirement is met."""
        # Implementation would check actual compliance status
        return "compliant" 