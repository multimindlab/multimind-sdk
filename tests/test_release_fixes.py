"""Regression tests for the 1.0.0 honesty fixes."""

import asyncio

import pytest

from multimind.compliance.ai_act import AIActCompliance
from multimind.compliance.governance import GovernanceConfig, RiskLevel
from multimind.fine_tuning.unified_fine_tuner import HyperparameterTuner
from multimind.mcp.advanced_executor import AdvancedMCPExecutor
from multimind.vector_store import get_backend_class


def test_executor_requires_a_real_registry():
    with pytest.raises(ValueError, match="model_registry"):
        AdvancedMCPExecutor()


def test_ai_act_does_not_pass_without_evidence():
    compliance = AIActCompliance(
        config=GovernanceConfig(
            organization_id="org",
            organization_name="Org",
            dpo_email="dpo@example.com",
        )
    )
    compliance.risk_assessments["sys"] = {
        "risk_level": RiskLevel.HIGH,
        "findings": [],
        "recommendations": [],
    }
    result = asyncio.run(compliance.validate_compliance("sys"))
    assert result["status"] == "unverifiable"
    assert all(item["status"] == "unverifiable" for item in result["requirements"])


def test_hyperparameter_tuner_does_not_invent_a_result():
    tuner = HyperparameterTuner(model_builder=lambda params: params, search_space={}, backend="ray")
    with pytest.raises(NotImplementedError):
        tuner.tune(lambda model: 1.0, n_trials=1)


def test_experimental_vector_backend_is_not_on_the_default_path():
    with pytest.raises(NotImplementedError, match="not on the default backend path"):
        get_backend_class("ClarifaiBackend")


async def test_faiss_search_honors_an_id_filter():
    pytest.importorskip("faiss")
    from multimind.vector_store.base import VectorStoreConfig, VectorStoreType
    from multimind.vector_store.faiss import FAISSBackend

    backend = FAISSBackend(
        VectorStoreConfig(store_type=VectorStoreType.FAISS, connection_params={"dimension": 2})
    )
    await backend.add_vectors(
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"label": "near"}, {"label": "far"}],
        documents=[{"text": "near"}, {"text": "far"}],
        ids=["near", "far"],
    )
    hits = await backend.search([0.0, 1.0], k=1, filter_criteria={"id": "near"})
    assert [hit.id for hit in hits] == ["near"]


def test_document_parser_rejects_a_non_json_response():
    pytest.importorskip("numpy")
    from multimind.document_processing.advanced_document_processor import _parse_model_json_list

    with pytest.raises(ValueError, match="JSON list"):
        _parse_model_json_list("no sections here", "section extraction")

    parsed = _parse_model_json_list('prefix [{"title": "A"}] suffix', "section extraction")
    assert parsed == [{"title": "A"}]
