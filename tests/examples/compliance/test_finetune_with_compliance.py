"""
Tests for finetune_with_compliance example.
"""

import pytest  # noqa: E402

pytest.importorskip("torch", reason="requires multimind-sdk[finetune]")

import pytest
import asyncio
import torch
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import tempfile
import shutil

# Add examples directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "examples" / "compliance"))

try:
    from finetune_with_compliance import CompliantModelTrainer
except ImportError:
    CompliantModelTrainer = None


@pytest.fixture
def mock_model():
    """Create a mock BERT model."""
    model = MagicMock()
    model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
    model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
    model.train.return_value = None
    model.eval.return_value = None
    
    # Mock forward pass
    mock_outputs = MagicMock()
    mock_outputs.loss = torch.tensor(0.5, requires_grad=True)
    mock_outputs.logits = torch.randn(2, 2)
    model.return_value = mock_outputs
    
    def forward(**kwargs):
        outputs = MagicMock()
        outputs.loss = torch.tensor(0.5, requires_grad=True)
        outputs.logits = torch.randn(kwargs.get("input_ids", torch.zeros(2, 10)).shape[0], 2)
        return outputs
    
    model.side_effect = forward
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.randint(0, 1000, (2, 10)),
        "attention_mask": torch.ones(2, 10)
    }
    
    def tokenize(texts, **kwargs):
        return {
            "input_ids": torch.randint(0, 1000, (len(texts), 10)),
            "attention_mask": torch.ones(len(texts), 10)
        }
    
    tokenizer.side_effect = tokenize
    return tokenizer


@pytest.fixture
def sample_train_data():
    """Sample training data."""
    return [
        {"text": "This is a positive example", "label": 1},
        {"text": "This is a negative example", "label": 0},
        {"text": "Another positive example", "label": 1},
        {"text": "Another negative example", "label": 0},
    ]


@pytest.fixture
def sample_val_data():
    """Sample validation data."""
    return [
        {"text": "Validation positive", "label": 1},
        {"text": "Validation negative", "label": 0},
    ]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.mark.skipif(CompliantModelTrainer is None, reason="CompliantModelTrainer not available")
class TestCompliantModelTrainer:
    """Test suite for CompliantModelTrainer."""
    
    @patch('finetune_with_compliance.AutoModelForSequenceClassification')
    @patch('finetune_with_compliance.AutoTokenizer')
    def test_initialization(self, mock_tokenizer_class, mock_model_class):
        """Test CompliantModelTrainer initialization."""
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
        mock_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = lambda texts, **kwargs: {
            "input_ids": torch.randint(0, 1000, (len(texts), 10)),
            "attention_mask": torch.ones(len(texts), 10)
        }
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        
        compliance_config = {
            "epsilon": 1.0,
            "rules": [
                {"name": "privacy", "threshold": 0.8},
                {"name": "fairness", "threshold": 0.9}
            ]
        }
        
        trainer = CompliantModelTrainer(
            model_name="bert-base-uncased",
            compliance_config=compliance_config
        )
        
        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert trainer.compliance_shard is not None
        assert trainer.privacy is not None
        assert trainer.watermarking is not None
        assert trainer.explainer is not None
        assert trainer.training_history == []
        assert trainer.compliance_history == []
    
    @patch('finetune_with_compliance.AutoModelForSequenceClassification')
    @patch('finetune_with_compliance.AutoTokenizer')
    @pytest.mark.asyncio
    async def test_finetune_basic(self, mock_tokenizer_class, mock_model_class, sample_train_data):
        """Test basic finetuning without validation data."""
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
        mock_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = lambda texts, **kwargs: {
            "input_ids": torch.randint(0, 1000, (len(texts), 10)),
            "attention_mask": torch.ones(len(texts), 10)
        }
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock model forward pass
        def model_forward(**kwargs):
            outputs = MagicMock()
            outputs.loss = torch.tensor(0.5, requires_grad=True)
            outputs.logits = torch.randn(kwargs.get("input_ids", torch.zeros(2, 10)).shape[0], 2)
            return outputs
        
        mock_model.side_effect = model_forward
        
        trainer = CompliantModelTrainer(
            model_name="bert-base-uncased",
            compliance_config={"epsilon": 1.0, "rules": []}
        )
        
        # Mock the model's __call__ method
        trainer.model = mock_model

        # Compliance verification fails closed without a real rule engine
        with pytest.raises(NotImplementedError):
            await trainer.finetune(
                train_data=sample_train_data,
                num_epochs=1,
                batch_size=2,
                learning_rate=2e-5
            )
    
    @patch('finetune_with_compliance.AutoModelForSequenceClassification')
    @patch('finetune_with_compliance.AutoTokenizer')
    @pytest.mark.asyncio
    async def test_finetune_with_validation(self, mock_tokenizer_class, mock_model_class, 
                                           sample_train_data, sample_val_data):
        """Test finetuning with validation data."""
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
        mock_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = lambda texts, **kwargs: {
            "input_ids": torch.randint(0, 1000, (len(texts), 10)),
            "attention_mask": torch.ones(len(texts), 10)
        }
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        
        def model_forward(**kwargs):
            outputs = MagicMock()
            outputs.loss = torch.tensor(0.5, requires_grad=True)
            batch_size = kwargs.get("input_ids", torch.zeros(2, 10)).shape[0]
            outputs.logits = torch.randn(batch_size, 2)
            return outputs
        
        mock_model.side_effect = model_forward
        
        trainer = CompliantModelTrainer(
            model_name="bert-base-uncased",
            compliance_config={"epsilon": 1.0, "rules": []}
        )
        trainer.model = mock_model

        # Compliance verification fails closed without a real rule engine
        with pytest.raises(NotImplementedError):
            await trainer.finetune(
                train_data=sample_train_data,
                val_data=sample_val_data,
                num_epochs=1,
                batch_size=2,
                learning_rate=2e-5
            )
    
    @patch('finetune_with_compliance.AutoModelForSequenceClassification')
    @patch('finetune_with_compliance.AutoTokenizer')
    @pytest.mark.asyncio
    async def test_private_forward(self, mock_tokenizer_class, mock_model_class):
        """Test private forward pass."""
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
        mock_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = lambda texts, **kwargs: {
            "input_ids": torch.randint(0, 1000, (len(texts), 10)),
            "attention_mask": torch.ones(len(texts), 10)
        }
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        
        def model_forward(**kwargs):
            outputs = MagicMock()
            outputs.loss = torch.tensor(0.5, requires_grad=True)
            outputs.logits = torch.randn(2, 2)
            return outputs
        
        mock_model.side_effect = model_forward
        
        trainer = CompliantModelTrainer(
            model_name="bert-base-uncased",
            compliance_config={"epsilon": 1.0, "rules": []}
        )
        trainer.model = mock_model
        
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.tensor([0, 1])
        }
        
        outputs = await trainer._private_forward(batch)
        
        assert outputs is not None
        assert hasattr(outputs, 'loss')
        assert hasattr(outputs, 'logits')
    
    @patch('finetune_with_compliance.AutoModelForSequenceClassification')
    @patch('finetune_with_compliance.AutoTokenizer')
    @pytest.mark.asyncio
    async def test_check_compliance(self, mock_tokenizer_class, mock_model_class):
        """Test compliance checking."""
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
        mock_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = lambda texts, **kwargs: {
            "input_ids": torch.randint(0, 1000, (len(texts), 10)),
            "attention_mask": torch.ones(len(texts), 10)
        }
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        
        trainer = CompliantModelTrainer(
            model_name="bert-base-uncased",
            compliance_config={"epsilon": 1.0, "rules": []}
        )
        
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.tensor([0, 1])
        }
        
        outputs = MagicMock()
        outputs.loss = torch.tensor(0.5)
        outputs.logits = torch.randn(2, 2)
        
        # Compliance verification fails closed without a real rule engine
        with pytest.raises(NotImplementedError):
            await trainer._check_compliance(batch, outputs)
    
    @patch('finetune_with_compliance.AutoModelForSequenceClassification')
    @patch('finetune_with_compliance.AutoTokenizer')
    @pytest.mark.asyncio
    async def test_check_epoch_compliance(self, mock_tokenizer_class, mock_model_class):
        """Test epoch compliance checking."""
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
        mock_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = lambda texts, **kwargs: {
            "input_ids": torch.randint(0, 1000, (len(texts), 10)),
            "attention_mask": torch.ones(len(texts), 10)
        }
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        
        trainer = CompliantModelTrainer(
            model_name="bert-base-uncased",
            compliance_config={"epsilon": 1.0, "rules": []}
        )
        
        # Explanation generation fails closed without a real explanation model
        with pytest.raises(NotImplementedError):
            await trainer._check_epoch_compliance(0.5)
    
    @patch('finetune_with_compliance.AutoModelForSequenceClassification')
    @patch('finetune_with_compliance.AutoTokenizer')
    @pytest.mark.asyncio
    async def test_validate(self, mock_tokenizer_class, mock_model_class, sample_val_data):
        """Test validation."""
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
        mock_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = lambda texts, **kwargs: {
            "input_ids": torch.randint(0, 1000, (len(texts), 10)),
            "attention_mask": torch.ones(len(texts), 10)
        }
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        
        def model_forward(**kwargs):
            outputs = MagicMock()
            outputs.loss = torch.tensor(0.5, requires_grad=True)
            batch_size = kwargs.get("input_ids", torch.zeros(2, 10)).shape[0]
            outputs.logits = torch.randn(batch_size, 2)
            return outputs
        
        mock_model.side_effect = model_forward
        
        trainer = CompliantModelTrainer(
            model_name="bert-base-uncased",
            compliance_config={"epsilon": 1.0, "rules": []}
        )
        trainer.model = mock_model
        
        val_dataloader = trainer._prepare_dataloader(sample_val_data, batch_size=2)
        metrics = await trainer._validate(val_dataloader)
        
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["accuracy"], float)
    
    @patch('finetune_with_compliance.AutoModelForSequenceClassification')
    @patch('finetune_with_compliance.AutoTokenizer')
    def test_prepare_dataloader(self, mock_tokenizer_class, mock_model_class, sample_train_data):
        """Test dataloader preparation."""
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
        mock_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = lambda texts, **kwargs: {
            "input_ids": torch.randint(0, 1000, (len(texts), 10)),
            "attention_mask": torch.ones(len(texts), 10)
        }
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        
        trainer = CompliantModelTrainer(
            model_name="bert-base-uncased",
            compliance_config={"epsilon": 1.0, "rules": []}
        )
        
        dataloader = trainer._prepare_dataloader(sample_train_data, batch_size=2)
        
        assert dataloader is not None
        assert dataloader.batch_size == 2
        
        # Test that batches are dictionaries
        for batch in dataloader:
            assert isinstance(batch, dict)
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "labels" in batch
            break
    
    @patch('finetune_with_compliance.AutoModelForSequenceClassification')
    @patch('finetune_with_compliance.AutoTokenizer')
    @pytest.mark.asyncio
    async def test_save_model(self, mock_tokenizer_class, mock_model_class, temp_dir):
        """Test model saving."""
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
        mock_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = lambda texts, **kwargs: {
            "input_ids": torch.randint(0, 1000, (len(texts), 10)),
            "attention_mask": torch.ones(len(texts), 10)
        }
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        
        trainer = CompliantModelTrainer(
            model_name="bert-base-uncased",
            compliance_config={"epsilon": 1.0, "rules": []}
        )
        
        # Add some training history
        trainer.training_history = [{"epoch": 0, "loss": 0.5}]
        trainer.compliance_history = [{"is_compliant": True, "compliance_score": 0.9}]
        
        model_path = os.path.join(temp_dir, "test_model.pt")
        # Watermarking fails closed without a real watermarking backend
        with pytest.raises(NotImplementedError):
            await trainer.save_model(model_path)
    
    @patch('finetune_with_compliance.AutoModelForSequenceClassification')
    @patch('finetune_with_compliance.AutoTokenizer')
    @pytest.mark.asyncio
    async def test_privacy_adaptation(self, mock_tokenizer_class, mock_model_class, sample_train_data):
        """Test privacy parameter adaptation during training."""
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
        mock_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = lambda texts, **kwargs: {
            "input_ids": torch.randint(0, 1000, (len(texts), 10)),
            "attention_mask": torch.ones(len(texts), 10)
        }
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        
        def model_forward(**kwargs):
            outputs = MagicMock()
            outputs.loss = torch.tensor(0.5, requires_grad=True)
            outputs.logits = torch.randn(kwargs.get("input_ids", torch.zeros(2, 10)).shape[0], 2)
            return outputs
        
        mock_model.side_effect = model_forward
        
        trainer = CompliantModelTrainer(
            model_name="bert-base-uncased",
            compliance_config={"epsilon": 1.0, "rules": []},
            privacy_config={
                "initial_epsilon": 1.0,
                "min_epsilon": 0.1,
                "max_epsilon": 10.0,
                "adaptation_rate": 0.1
            }
        )
        trainer.model = mock_model

        # Compliance verification fails closed before privacy adaptation runs
        with pytest.raises(NotImplementedError):
            await trainer.finetune(
                train_data=sample_train_data,
                num_epochs=1,
                batch_size=2,
                learning_rate=2e-5
            )


@pytest.mark.skipif(CompliantModelTrainer is None, reason="CompliantModelTrainer not available")
@pytest.mark.asyncio
async def test_integration_basic_training():
    """Integration test for basic training workflow."""
    with patch('finetune_with_compliance.AutoModelForSequenceClassification') as mock_model_class, \
         patch('finetune_with_compliance.AutoTokenizer') as mock_tokenizer:
        
        # Setup mocks
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
        mock_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        def model_forward(**kwargs):
            outputs = MagicMock()
            outputs.loss = torch.tensor(0.5, requires_grad=True)
            batch_size = kwargs.get("input_ids", torch.zeros(2, 10)).shape[0]
            outputs.logits = torch.randn(batch_size, 2)
            return outputs
        
        mock_model.side_effect = model_forward
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = lambda texts, **kwargs: {
            "input_ids": torch.randint(0, 1000, (len(texts), 10)),
            "attention_mask": torch.ones(len(texts), 10)
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create trainer and run training
        trainer = CompliantModelTrainer(
            model_name="bert-base-uncased",
            compliance_config={"epsilon": 1.0, "rules": []}
        )
        trainer.model = mock_model
        
        train_data = [
            {"text": "Positive example", "label": 1},
            {"text": "Negative example", "label": 0},
        ]
        
        # Compliance verification fails closed without a real rule engine
        with pytest.raises(NotImplementedError):
            await trainer.finetune(
                train_data=train_data,
                num_epochs=1,
                batch_size=2,
                learning_rate=2e-5
            )

