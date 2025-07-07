from multimind.client.model_client import ImageModelClient, AudioModelClient, CodeModelClient

def test_image_model_client():
    client = ImageModelClient()
    result = client.generate("prompt")
    assert "Placeholder image" in result

def test_audio_model_client():
    client = AudioModelClient()
    result = client.generate("prompt")
    assert "Placeholder audio" in result

def test_code_model_client():
    client = CodeModelClient()
    result = client.generate("prompt")
    assert "Placeholder code" in result 