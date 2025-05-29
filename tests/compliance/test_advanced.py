"""
Tests for advanced compliance features.
"""

import pytest
import torch
import numpy as np
from datetime import datetime
from multimind.compliance.advanced import (
    ComplianceShard,
    SelfHealingCompliance,
    ExplainableDTO,
    ModelWatermarking,
    AdaptivePrivacy,
    RegulatoryChangeDetector,
    FederatedCompliance,
    ComplianceLevel,
    ComplianceMetrics
)
from multimind.compliance.advanced_config import (
    ComplianceShardConfig,
    SelfHealingConfig,
    ExplainableDTOConfig,
    ModelWatermarkingConfig,
    AdaptivePrivacyConfig,
    RegulatoryChangeConfig,
    FederatedComplianceConfig,
    ComplianceLevel as ConfigComplianceLevel,
    ConsensusMethod
)

@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2)
    )

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return torch.randn(100, 10)

@pytest.fixture
def shard_config():
    """Create a sample shard configuration."""
    return ComplianceShardConfig(
        shard_id="test_shard",
        jurisdiction="test_jurisdiction",
        epsilon=1.0,
        rules=[{"name": "test_rule", "threshold": 0.8}],
        metadata={"description": "Test shard"},
        compliance_level=ConfigComplianceLevel.ADVANCED,
        encryption_enabled=True,
        metrics_tracking=True,
        resource_limits={"cpu": 1.0, "memory": 1024.0, "network": 100.0}
    )

@pytest.fixture
def self_healing_config():
    """Create a sample self-healing configuration."""
    return SelfHealingConfig(
        auto_patch=True,
        rollback_enabled=True,
        notification_channels=["test"],
        vulnerability_threshold=0.8,
        patch_history_size=10,
        effectiveness_tracking=True,
        rollback_points=5,
        patch_validation=True,
        impact_analysis=True
    )

@pytest.fixture
def explainable_dto_config():
    """Create a sample explainable DTO configuration."""
    return ExplainableDTOConfig(
        model_version="test",
        confidence_threshold=0.8,
        explanation_depth=2,
        include_metadata=True,
        uncertainty_estimation=True,
        factor_importance=True,
        explanation_history=True,
        visualization_enabled=True
    )

@pytest.fixture
def watermarking_config():
    """Create a sample watermarking configuration."""
    return ModelWatermarkingConfig(
        watermark_type="invisible",
        fingerprint_size=128,
        tracking_enabled=True,
        verification_threshold=0.9,
        tamper_detection=True,
        version_tracking=True,
        verification_history=True,
        security_level="high"
    )

@pytest.fixture
def adaptive_privacy_config():
    """Create a sample adaptive privacy configuration."""
    return AdaptivePrivacyConfig(
        initial_epsilon=1.0,
        min_epsilon=0.1,
        max_epsilon=10.0,
        adaptation_rate=0.1,
        feedback_window=10,
        adaptation_strategy="dynamic",
        privacy_metrics=True,
        validation_enabled=True,
        guarantees_verification=True
    )

@pytest.fixture
def regulatory_config():
    """Create a sample regulatory configuration."""
    return RegulatoryChangeConfig(
        sources=[{"name": "test", "url": "http://test.com"}],
        check_interval=1,
        auto_patch=True,
        notification_channels=["test"],
        impact_analysis=True,
        patch_validation=True,
        patch_testing=True,
        change_history=True
    )

@pytest.fixture
def federated_config():
    """Create a sample federated configuration."""
    return FederatedComplianceConfig(
        shards=[shard_config()],
        coordinator={"type": "test"},
        aggregation_method="weighted",
        proof_generation=True,
        consensus_method=ConsensusMethod.WEIGHTED,
        load_balancing=True,
        verification_history=True,
        security_level="high"
    )

def test_compliance_shard(shard_config, sample_model, sample_data):
    """Test enhanced compliance shard functionality."""
    shard = ComplianceShard(shard_config)
    
    # Test compliance verification with different levels
    for level in ComplianceLevel:
        result = shard.verify_compliance(sample_model, sample_data, level)
        assert isinstance(result, dict)
        assert "compliance_score" in result
        assert "violations" in result
        assert "metrics" in result
    
    # Test metrics calculation
    metrics = result["metrics"]
    assert isinstance(metrics, ComplianceMetrics)
    assert 0 <= metrics.score <= 1
    assert 0 <= metrics.confidence <= 1
    assert isinstance(metrics.risk_level, str)
    assert metrics.verification_time > 0
    assert isinstance(metrics.resource_usage, dict)
    
    # Test encryption
    assert "encrypted_result" in result

def test_self_healing_compliance(self_healing_config, sample_model):
    """Test enhanced self-healing compliance functionality."""
    healing = SelfHealingCompliance(self_healing_config)
    
    # Test vulnerability check with severity assessment
    vulnerabilities = healing.check_vulnerabilities(sample_model)
    assert isinstance(vulnerabilities, list)
    for vuln in vulnerabilities:
        assert "severity" in vuln
        assert "impact" in vuln
    
    # Test patch application with effectiveness tracking
    patched_model = healing.apply_patch(sample_model, vulnerabilities[0])
    assert isinstance(patched_model, torch.nn.Module)
    assert healing.patch_effectiveness
    
    # Test rollback with history
    healing.rollback_patch(sample_model)
    assert healing.rollback_points

def test_explainable_dto(explainable_dto_config, sample_model, sample_data):
    """Test enhanced explainable DTO functionality."""
    dto = ExplainableDTO(explainable_dto_config)
    
    # Test explanation generation with different depths
    for depth in range(1, 4):
        explanation = dto.generate_explanation(sample_model, sample_data, depth)
        assert isinstance(explanation, dict)
        assert "explanation" in explanation
        assert "confidence" in explanation
        assert "uncertainty" in explanation
        assert "factor_importance" in explanation
    
    # Test explanation history
    assert dto.explanation_history

def test_model_watermarking(watermarking_config, sample_model):
    """Test enhanced model watermarking functionality."""
    watermarking = ModelWatermarking(watermarking_config)
    
    # Test watermark application with tamper detection
    watermarked_model = watermarking.apply_watermark(sample_model)
    assert isinstance(watermarked_model, torch.nn.Module)
    
    # Test fingerprint tracking with versioning
    fingerprint = watermarking.track_fingerprint(watermarked_model)
    assert isinstance(fingerprint, dict)
    assert "fingerprint" in fingerprint
    assert "version" in fingerprint
    
    # Test verification with tamper detection
    verification_result = watermarking.verify_watermark(watermarked_model)
    assert isinstance(verification_result, dict)
    assert "is_valid" in verification_result
    assert "confidence" in verification_result
    assert "tamper_detected" in verification_result
    assert "tamper_details" in verification_result
    
    # Test verification history
    assert watermarking.verification_history

@pytest.mark.asyncio
async def test_adaptive_privacy(adaptive_privacy_config, sample_model, sample_data):
    """Test enhanced adaptive privacy functionality."""
    privacy = AdaptivePrivacy(adaptive_privacy_config)
    
    # Test privacy adaptation with strategy
    adapted_model = privacy.adapt_privacy(sample_model, sample_data)
    assert isinstance(adapted_model, torch.nn.Module)
    
    # Test feedback incorporation with metrics
    privacy.incorporate_feedback(adapted_model, 0.8)
    assert privacy.privacy_metrics
    
    # Test validation and guarantees
    assert privacy._validate_epsilon(privacy.dp_mechanism.epsilon)
    await privacy._verify_privacy_guarantees()

def test_regulatory_change_detector(regulatory_config, sample_model):
    """Test enhanced regulatory change detector functionality."""
    detector = RegulatoryChangeDetector(regulatory_config)
    
    # Test change detection with impact analysis
    changes = detector.detect_changes()
    assert isinstance(changes, list)
    for change in changes:
        assert "impact" in change
    
    # Test patch generation with validation and testing
    if changes:
        patch = detector.generate_patch(sample_model, changes[0])
        assert isinstance(patch, dict)
        assert "patch" in patch
        assert "validation" in patch
        assert "test_results" in patch

def test_federated_compliance(federated_config, sample_model, sample_data):
    """Test enhanced federated compliance functionality."""
    federated = FederatedCompliance(federated_config)
    
    # Test compliance verification with consensus
    result = federated.verify_global_compliance(sample_model, sample_data)
    assert isinstance(result, dict)
    assert "compliance_score" in result
    assert "jurisdiction_results" in result
    assert "consensus" in result
    
    # Test proof generation with enhanced security
    proof = federated.generate_proof(sample_model, sample_data)
    assert isinstance(proof, dict)
    assert "proof" in proof
    assert "consensus_evidence" in proof
    assert "signature" in proof
    
    # Test verification history
    assert federated.verification_history 