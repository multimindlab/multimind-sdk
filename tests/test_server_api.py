import pytest
from multimind.server import MultiMindServer
from multimind.api import multi_model_app, unified_app

def test_multimind_server_init():
    server = MultiMindServer()
    assert server is not None

def test_multi_model_app_exists():
    assert multi_model_app is not None

def test_unified_app_exists():
    assert unified_app is not None 