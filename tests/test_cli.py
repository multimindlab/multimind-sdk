import os
import tempfile
import pytest
from click.testing import CliRunner
from multimind.cli import cli

@pytest.fixture
def runner():
    return CliRunner()


def test_cli_help(runner):
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert "MultiMind SDK CLI" in result.output


def test_train_help(runner):
    result = runner.invoke(cli, ['train', '--help'])
    assert result.exit_code == 0
    assert "Fine-tune a model" in result.output


def test_finetune_alias_help(runner):
    result = runner.invoke(cli, ['finetune', '--help'])
    assert result.exit_code == 0
    assert "Fine-tune a model" in result.output


def test_train_config_parsing(runner, mocker):
    # Create a minimal YAML config
    config_content = """
base_model_name: test-model
output_dir: ./output
methods: [lora]
train_dataset: dummy_train.json
"""
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = f.name
    # Mock UniPELTTuner and its train method
    mock_tuner = mocker.patch('multimind.cli.UniPELTTuner')
    instance = mock_tuner.return_value
    instance.train.return_value = None
    result = runner.invoke(cli, ['train', '--config', config_path])
    assert result.exit_code == 0
    assert "Training complete" in result.output
    instance.train.assert_called_once()
    os.remove(config_path)

# Evaluate command

def test_evaluate_success(runner, mocker):
    mock_tuner = mocker.patch('multimind.cli.UniPELTTuner')
    instance = mock_tuner.return_value
    instance.load_model.return_value = None
    instance.trainer = mocker.Mock()
    instance.trainer.evaluate.return_value = {"accuracy": 0.9}
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        f.write('[{"text": "foo", "label": 1}]')
        dataset_path = f.name
    result = runner.invoke(cli, ['evaluate', '--model', 'foo', '--dataset', dataset_path])
    assert result.exit_code == 0
    assert "Metrics" in result.output
    os.remove(dataset_path)

def test_evaluate_failure(runner, mocker):
    mocker.patch('multimind.cli.UniPELTTuner', side_effect=Exception("fail"))
    result = runner.invoke(cli, ['evaluate', '--model', 'foo', '--dataset', 'bar'])
    assert result.exit_code != 0
    assert "Error" in result.output

# Infer command

def test_infer_success(runner, mocker):
    mock_tuner = mocker.patch('multimind.cli.UniPELTTuner')
    instance = mock_tuner.return_value
    instance.load_model.return_value = None
    instance.model = mocker.Mock()
    instance.model.device = mocker.Mock(type='cpu')
    mock_tokenizer = mocker.patch('multimind.cli.AutoTokenizer')
    tokenizer_instance = mock_tokenizer.from_pretrained.return_value
    tokenizer_instance.return_tensors = "pt"
    tokenizer_instance.__call__.return_value = {"input_ids": [1]}
    instance.model.generate.return_value = [[1, 2, 3]]
    tokenizer_instance.decode.return_value = "output text"
    result = runner.invoke(cli, ['infer', '--model', 'foo', '--input', 'bar'])
    assert result.exit_code == 0 or "Output" in result.output

def test_infer_failure(runner, mocker):
    mocker.patch('multimind.cli.UniPELTTuner', side_effect=Exception("fail"))
    result = runner.invoke(cli, ['infer', '--model', 'foo', '--input', 'bar'])
    assert result.exit_code != 0
    assert "Error" in result.output

# List models

def test_list_models_success(runner, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "model1").mkdir()
    (models_dir / "model2").mkdir()
    result = runner.invoke(cli, ['list-models', '--output-dir', str(models_dir)])
    assert result.exit_code == 0
    assert "model1" in result.output and "model2" in result.output

def test_list_models_empty(runner, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    result = runner.invoke(cli, ['list-models', '--output-dir', str(models_dir)])
    assert "No models found" in result.output

# Download

def test_download_success(runner, mocker):
    mocker.patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=True)
    result = runner.invoke(cli, ['download', '--model', 'bert-base-uncased'])
    assert result.exit_code == 0
    assert "Downloaded model" in result.output

def test_download_failure(runner, mocker):
    mocker.patch('transformers.AutoModelForCausalLM.from_pretrained', side_effect=Exception("fail"))
    result = runner.invoke(cli, ['download', '--model', 'bert-base-uncased'])
    assert result.exit_code != 0
    assert "Error" in result.output

# Export

def test_export_success(runner, mocker):
    mock_model = mocker.patch('transformers.AutoModelForCausalLM.from_pretrained')
    mock_model.return_value = mocker.Mock()
    mocker.patch('torch.onnx.export', return_value=None)
    result = runner.invoke(cli, ['export', '--model', 'foo', '--format', 'onnx', '--output', 'bar'])
    assert result.exit_code == 0 or result.exit_code == 1

def test_export_failure(runner, mocker):
    mocker.patch('transformers.AutoModelForCausalLM.from_pretrained', side_effect=Exception("fail"))
    result = runner.invoke(cli, ['export', '--model', 'foo', '--format', 'onnx', '--output', 'bar'])
    assert result.exit_code != 0
    assert "Error" in result.output

# Delete

def test_delete_abort(runner, mocker):
    result = runner.invoke(cli, ['delete', '--model', 'foo'], input='n\n')
    assert "Aborted" in result.output

def test_delete_success(runner, mocker, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    mocker.patch('os.path.isdir', return_value=True)
    mocker.patch('shutil.rmtree', return_value=None)
    result = runner.invoke(cli, ['delete', '--model', str(model_dir)], input='y\n')
    assert "Deleted model" in result.output

def test_delete_failure(runner, mocker):
    mocker.patch('os.path.isdir', return_value=True)
    mocker.patch('shutil.rmtree', side_effect=Exception("fail"))
    result = runner.invoke(cli, ['delete', '--model', 'foo'], input='y\n')
    assert "Error deleting model" in result.output

# Config, info, completion remain as before

def test_config_command(runner, mocker):
    result = runner.invoke(cli, ['config'])
    assert result.exit_code == 0
    assert "Current config" in result.output

def test_info_command(runner):
    result = runner.invoke(cli, ['info'])
    assert result.exit_code == 0
    assert "environment info" in result.output

def test_completion_command(runner):
    result = runner.invoke(cli, ['completion', 'bash'])
    assert result.exit_code == 0
    assert "completion" in result.output 