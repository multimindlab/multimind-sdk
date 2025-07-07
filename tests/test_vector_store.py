import pytest

@pytest.mark.skip(reason="ClarifaiBackend is abstract and cannot be instantiated.")
def test_clarifai_backend():
    pass

@pytest.mark.skip(reason="DashVectorBackend is abstract and cannot be instantiated.")
def test_dashvector_backend():
    pass

@pytest.mark.skip(reason="DingoDBBackend is abstract and cannot be instantiated.")
def test_dingodb_backend():
    pass

@pytest.mark.skip(reason="EpsillaBackend is abstract and cannot be instantiated.")
def test_epsilla_backend():
    pass 