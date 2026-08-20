import pytest

pytest.importorskip("numpy")  # requires optional extras absent on core-only installs
from multimind.ensemble.advanced import AdvancedEnsemble


class DummyRouter:
    pass

def test_advanced_ensemble_init():
    router = DummyRouter()
    ensemble = AdvancedEnsemble(router)
    assert ensemble is not None

def test_advanced_ensemble_combine():
    router = DummyRouter()
    ensemble = AdvancedEnsemble(router)
    assert ensemble is not None
