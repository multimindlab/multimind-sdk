"""
Compliance policy engine implementation.
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import logging
import uuid
from pydantic import BaseModel, Field
from .governance import GovernanceConfig, Regulation, RiskLevel

logger = logging.getLogger(__name__)

class PolicyRule(BaseModel):
    """Policy rule model."""
    
    rule_id: str
    name: str
    description: str
    regulation: Regulation
    risk_level: Optional[RiskLevel] = None
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    enabled: bool = True
    priority: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PolicyViolation(BaseModel):
    """Policy violation model."""
    
    violation_id: str
    rule_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Dict[str, Any] = Field(default_factory=dict)
    severity: str
    status: str = "open"
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None

class CompliancePolicyEngine(BaseModel):
    """Compliance policy engine."""
    
    config: GovernanceConfig
    rules: Dict[str, PolicyRule] = Field(default_factory=dict)
    violations: List[PolicyViolation] = Field(default_factory=list)
    rule_handlers: Dict[str, Callable] = Field(default_factory=dict)
    
    async def add_rule(self, rule: PolicyRule) -> None:
        """Add a policy rule."""
        self.rules[rule.rule_id] = rule
    
    async def remove_rule(self, rule_id: str) -> None:
        """Remove a policy rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
    
    async def register_handler(
        self,
        rule_id: str,
        handler: Callable
    ) -> None:
        """Register a handler for a rule."""
        self.rule_handlers[rule_id] = handler
    
    async def evaluate_policy(
        self,
        context: Dict[str, Any]
    ) -> List[PolicyViolation]:
        """Evaluate policy rules against context."""
        violations = []
        
        # Sort rules by priority
        sorted_rules = sorted(
            self.rules.values(),
            key=lambda r: r.priority,
            reverse=True
        )
        
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            
            # Check if rule applies to context
            if not self._rule_applies(rule, context):
                continue
            
            # Evaluate rule conditions
            if not self._evaluate_conditions(rule, context):
                # Create violation
                violation = PolicyViolation(
                    violation_id=f"viol_{uuid.uuid4()}",
                    rule_id=rule.rule_id,
                    details={
                        "context": context,
                        "rule": rule.dict()
                    },
                    severity=self._determine_severity(rule)
                )
                
                violations.append(violation)
                self.violations.append(violation)
                
                # Execute rule actions
                await self._execute_actions(rule, context, violation)
        
        return violations
    
    async def resolve_violation(
        self,
        violation_id: str,
        resolution: str
    ) -> Optional[PolicyViolation]:
        """Resolve a policy violation."""
        for violation in self.violations:
            if violation.violation_id == violation_id:
                violation.status = "resolved"
                violation.resolution = resolution
                violation.resolved_at = datetime.now()
                return violation
        return None
    
    async def get_active_violations(
        self,
        rule_id: Optional[str] = None,
        severity: Optional[str] = None
    ) -> List[PolicyViolation]:
        """Get active policy violations."""
        violations = [v for v in self.violations if v.status == "open"]
        
        if rule_id:
            violations = [v for v in violations if v.rule_id == rule_id]
        if severity:
            violations = [v for v in violations if v.severity == severity]
        
        return violations
    
    def _rule_applies(self, rule: PolicyRule, context: Dict[str, Any]) -> bool:
        """Check if rule applies to context."""
        # Check regulation
        if rule.regulation not in self.config.enabled_regulations:
            return False
        
        # Check risk level
        if rule.risk_level and context.get("risk_level") != rule.risk_level:
            return False
        
        return True
    
    def _evaluate_conditions(
        self,
        rule: PolicyRule,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate rule conditions."""
        for condition in rule.conditions:
            if not self._evaluate_condition(condition, context):
                return False
        return True
    
    def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate a single condition."""
        field = condition.get("field")
        operator = condition.get("operator")
        value = condition.get("value")
        
        # `value` may validly be falsy (e.g., 0, False, ""), so only field/operator are required.
        if not field or not operator or "value" not in condition:
            return False
        
        context_value = context.get(field)
        
        if operator == "equals":
            return context_value == value
        elif operator == "not_equals":
            return context_value != value
        elif operator == "contains":
            if context_value is None:
                return False
            return value in context_value
        elif operator == "not_contains":
            if context_value is None:
                return True
            return value not in context_value
        elif operator == "greater_than":
            if context_value is None:
                return False
            return context_value > value
        elif operator == "less_than":
            if context_value is None:
                return False
            return context_value < value
        elif operator == "in":
            if value is None:
                return False
            return context_value in value
        elif operator == "not_in":
            if value is None:
                return True
            return context_value not in value
        
        return False
    
    async def _execute_actions(
        self,
        rule: PolicyRule,
        context: Dict[str, Any],
        violation: PolicyViolation
    ) -> None:
        """Execute rule actions."""
        for action in rule.actions:
            action_type = action.get("type")
            action_params = action.get("params", {})
            
            if action_type == "log":
                logger.warning(
                    "[COMPLIANCE LOG] Violation: %s | Rule: %s | Details: %s",
                    violation.violation_id,
                    rule.name,
                    violation.details,
                )
            elif action_type == "notify":
                # Simulate sending a notification (could be email, webhook, etc.)
                recipient = action_params.get("recipient", "admin")
                message = action_params.get("message", f"Policy violation: {violation.violation_id}")
                logger.info("[COMPLIANCE NOTIFY] To: %s | Message: %s", recipient, message)
            elif action_type == "block":
                # Block operation by raising an exception
                raise Exception(f"Operation blocked due to policy violation: {violation.violation_id} (Rule: {rule.name})")
            elif action_type == "custom":
                # Execute custom handler
                handler = self.rule_handlers.get(rule.rule_id)
                if handler:
                    await handler(context, violation, action_params)
    
    def _determine_severity(self, rule: PolicyRule) -> str:
        """Determine violation severity."""
        if rule.risk_level == RiskLevel.HIGH:
            return "high"
        elif rule.risk_level == RiskLevel.LIMITED:
            return "medium"
        return "low" 