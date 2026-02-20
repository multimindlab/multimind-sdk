"""
Tests for the multi-modal basic process_request example.

These tests call the FastAPI app directly (no uvicorn) and monkeypatch
expert building to avoid external provider/API calls.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Import inside fixture so monkeypatching affects the module instance used by the app.
    import multimind.api.unified_api as unified_api
    from multimind.models.moe import Expert

    class TextExpertStub(Expert):
        async def process(self, input_data):
            if isinstance(input_data, dict):
                text = input_data.get("text", "")
            else:
                text = input_data
            return f"TEXT({text})"

    class ImageExpertStub(Expert):
        async def process(self, input_data):
            return "IMAGE_OK"

    class AudioExpertStub(Expert):
        async def process(self, input_data):
            return "AUDIO_OK"

    def fake_build_experts(modalities, router):
        experts = {}
        if "text" in modalities:
            experts["text_expert"] = TextExpertStub("text_expert")
        if "image" in modalities:
            experts["image_expert"] = ImageExpertStub("image_expert")
        if "audio" in modalities:
            experts["audio_expert"] = AudioExpertStub("audio_expert")
        return experts

    monkeypatch.setattr(unified_api, "_build_experts", fake_build_experts)
    return TestClient(unified_api.app)


@pytest.mark.skipif(not _torch_available(), reason="MoE requires torch for these tests")
def test_image_plus_text_weights_are_half_half(client: TestClient):
    payload = {
        "inputs": [
            {"modality": "image", "content": "aW1hZ2U="},
            {"modality": "text", "content": "Describe the image"},
        ],
        "use_moe": True,
    }
    resp = client.post("/v1/process", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    weights = data["expert_weights"]
    assert weights["image_expert"] == pytest.approx(0.5, abs=1e-6)
    assert weights["text_expert"] == pytest.approx(0.5, abs=1e-6)
    assert "text" in data["outputs"]
    assert data["outputs"].get("image_text") == "IMAGE_OK"


@pytest.mark.skipif(not _torch_available(), reason="MoE requires torch for these tests")
def test_audio_plus_text_weights_are_half_half(client: TestClient):
    payload = {
        "inputs": [
            {"modality": "audio", "content": "YXVkaW8="},
            {"modality": "text", "content": "Transcribe and summarize"},
        ],
        "use_moe": True,
    }
    resp = client.post("/v1/process", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    weights = data["expert_weights"]
    assert weights["audio_expert"] == pytest.approx(0.5, abs=1e-6)
    assert weights["text_expert"] == pytest.approx(0.5, abs=1e-6)
    assert "text" in data["outputs"]
    assert data["outputs"].get("audio_text") == "AUDIO_OK"


@pytest.mark.skipif(not _torch_available(), reason="MoE requires torch for these tests")
def test_text_image_audio_weights_are_thirds(client: TestClient):
    payload = {
        "inputs": [
            {"modality": "image", "content": "aW1hZ2U="},
            {"modality": "audio", "content": "YXVkaW8="},
            {"modality": "text", "content": "Analyze both"},
        ],
        "use_moe": True,
    }
    resp = client.post("/v1/process", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    weights = data["expert_weights"]
    assert weights["text_expert"] == pytest.approx(1 / 3, abs=1e-6)
    assert weights["image_expert"] == pytest.approx(1 / 3, abs=1e-6)
    assert weights["audio_expert"] == pytest.approx(1 / 3, abs=1e-6)
    assert data["outputs"].get("image_text") == "IMAGE_OK"
    assert data["outputs"].get("audio_text") == "AUDIO_OK"
    assert "text" in data["outputs"]


@pytest.mark.skipif(not _torch_available(), reason="MoE requires torch for these tests")
def test_text_only_weight_is_one(client: TestClient):
    payload = {
        "inputs": [
            {"modality": "text", "content": "Hello"},
        ],
        "use_moe": True,
    }
    resp = client.post("/v1/process", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    weights = data["expert_weights"]
    assert weights["text_expert"] == pytest.approx(1.0, abs=1e-6)
    assert "text" in data["outputs"]
    assert "image_text" not in data["outputs"]
    assert "audio_text" not in data["outputs"]


@pytest.mark.skipif(not _torch_available(), reason="MoE requires torch for these tests")
def test_unknown_modality_returns_400(client: TestClient):
    payload = {
        "inputs": [
            {"modality": "video", "content": "Zm9v"},
        ],
        "use_moe": True,
    }
    resp = client.post("/v1/process", json=payload)
    assert resp.status_code == 400, resp.text
