"""
Example demonstrating custom training data preparation utilities and model version management features.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from multimind.models.factory import ModelFactory
from multimind.models.multi_model import MultiModelWrapper

class DataFormat(Enum):
    """Supported training data formats."""
    JSON = "json"
    CSV = "csv"
    TXT = "txt"
    JSONL = "jsonl"

@dataclass
class TrainingExample:
    """Represents a single training example."""
    prompt: str
    expected_response: str
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

class TrainingDataManager:
    """Manages training data preparation and validation."""
    
    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.examples: List[TrainingExample] = []
        self.validation_examples: List[TrainingExample] = []
        
    def load_data(self, file_path: str, format: DataFormat) -> List[TrainingExample]:
        """Load training data from various formats."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Training data file not found: {file_path}")
            
        if format == DataFormat.JSON:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return [TrainingExample(**example) for example in data]
                
        elif format == DataFormat.CSV:
            df = pd.read_csv(file_path)
            return [
                TrainingExample(
                    prompt=row['prompt'],
                    expected_response=row['expected_response'],
                    metadata=row.get('metadata'),
                    tags=row.get('tags', '').split(',') if 'tags' in row else None
                )
                for _, row in df.iterrows()
            ]
            
        elif format == DataFormat.JSONL:
            examples = []
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    examples.append(TrainingExample(**data))
            return examples
            
        else:
            raise ValueError(f"Unsupported data format: {format}")
    
    def prepare_training_data(
        self,
        examples: List[TrainingExample],
        validation_split: float = 0.1,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ) -> Dict[str, List[TrainingExample]]:
        """Prepare training and validation datasets."""
        if shuffle:
            train_examples, val_examples = train_test_split(
                examples,
                test_size=validation_split,
                random_state=random_state
            )
        else:
            split_idx = int(len(examples) * (1 - validation_split))
            train_examples = examples[:split_idx]
            val_examples = examples[split_idx:]
            
        self.examples = train_examples
        self.validation_examples = val_examples
        
        return {
            'training': train_examples,
            'validation': val_examples
        }
    
    def validate_examples(self) -> Dict[str, Any]:
        """Validate training examples for quality and consistency."""
        stats = {
            'total_examples': len(self.examples),
            'validation_examples': len(self.validation_examples),
            'avg_prompt_length': 0,
            'avg_response_length': 0,
            'has_metadata': 0,
            'has_tags': 0
        }
        
        if not self.examples:
            return stats
            
        total_prompt_length = sum(len(ex.prompt) for ex in self.examples)
        total_response_length = sum(len(ex.expected_response) for ex in self.examples)
        
        stats['avg_prompt_length'] = total_prompt_length / len(self.examples)
        stats['avg_response_length'] = total_response_length / len(self.examples)
        stats['has_metadata'] = sum(1 for ex in self.examples if ex.metadata)
        stats['has_tags'] = sum(1 for ex in self.examples if ex.tags)
        
        return stats
    
    def save_data(self, file_path: str, format: DataFormat):
        """Save training data to file."""
        file_path = Path(file_path)
        
        if format == DataFormat.JSON:
            data = [vars(ex) for ex in self.examples]
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        elif format == DataFormat.CSV:
            df = pd.DataFrame([vars(ex) for ex in self.examples])
            df.to_csv(file_path, index=False)
            
        elif format == DataFormat.JSONL:
            with open(file_path, 'w') as f:
                for ex in self.examples:
                    f.write(json.dumps(vars(ex)) + '\n')
                    
        else:
            raise ValueError(f"Unsupported data format: {format}")

class ModelVersionManager:
    """Manages model versions and their metadata."""
    
    def __init__(self, version_dir: str = "model_versions"):
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(exist_ok=True)
        self.versions: Dict[str, Dict[str, Any]] = {}
        self.load_versions()
    
    def load_versions(self):
        """Load existing model versions from disk."""
        version_file = self.version_dir / "versions.json"
        if version_file.exists():
            with open(version_file, 'r') as f:
                self.versions = json.load(f)
    
    def save_versions(self):
        """Save model versions to disk."""
        version_file = self.version_dir / "versions.json"
        with open(version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def create_version(
        self,
        model_name: str,
        base_model: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new model version."""
        version_id = f"{model_name}_v{len(self.versions) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.versions[version_id] = {
            'model_name': model_name,
            'base_model': base_model,
            'config': config,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'status': 'created',
            'performance_metrics': {},
            'training_history': []
        }
        
        self.save_versions()
        return version_id
    
    def update_version(
        self,
        version_id: str,
        updates: Dict[str, Any]
    ):
        """Update model version metadata."""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
            
        self.versions[version_id].update(updates)
        self.versions[version_id]['updated_at'] = datetime.now().isoformat()
        self.save_versions()
    
    def add_training_record(
        self,
        version_id: str,
        metrics: Dict[str, Any],
        epoch: int
    ):
        """Add a training record to version history."""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
            
        record = {
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.versions[version_id]['training_history'].append(record)
        self.save_versions()
    
    def get_version(self, version_id: str) -> Dict[str, Any]:
        """Get version information."""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        return self.versions[version_id]
    
    def list_versions(
        self,
        model_name: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List model versions with optional filtering."""
        versions = []
        for version_id, version_data in self.versions.items():
            if model_name and version_data['model_name'] != model_name:
                continue
            if status and version_data['status'] != status:
                continue
            versions.append({
                'version_id': version_id,
                **version_data
            })
        return versions
    
    def compare_versions(
        self,
        version_ids: List[str],
        metric: str = 'accuracy'
    ) -> Dict[str, Any]:
        """Compare multiple versions based on a specific metric."""
        comparison = {}
        for version_id in version_ids:
            if version_id not in self.versions:
                raise ValueError(f"Version {version_id} not found")
                
            version_data = self.versions[version_id]
            metrics = version_data.get('performance_metrics', {})
            
            comparison[version_id] = {
                'metric': metrics.get(metric),
                'model_name': version_data['model_name'],
                'created_at': version_data['created_at']
            }
            
        return comparison

async def run_training_utils_examples():
    # Initialize managers
    data_manager = TrainingDataManager()
    version_manager = ModelVersionManager()
    
    # Example 1: Load and prepare training data
    print("Example 1: Loading and preparing training data")
    examples = [
        TrainingExample(
            prompt="What is the capital of France?",
            expected_response="Paris",
            tags=["geography", "capital"]
        ),
        TrainingExample(
            prompt="Who wrote Romeo and Juliet?",
            expected_response="William Shakespeare",
            tags=["literature", "author"]
        )
    ]
    
    prepared_data = data_manager.prepare_training_data(examples)
    validation_stats = data_manager.validate_examples()
    print("\nValidation stats:")
    print(json.dumps(validation_stats, indent=2))
    
    # Example 2: Create and manage model versions
    print("\nExample 2: Model version management")
    version_id = version_manager.create_version(
        model_name="qa_model",
        base_model="openai",
        config={
            "temperature": 0.7,
            "max_tokens": 100
        },
        metadata={
            "description": "Question answering model",
            "tags": ["qa", "general"]
        }
    )
    
    # Add training records
    for epoch in range(3):
        metrics = {
            "accuracy": 0.8 + epoch * 0.05,
            "loss": 0.2 - epoch * 0.05
        }
        version_manager.add_training_record(version_id, metrics, epoch)
    
    # Update version status
    version_manager.update_version(version_id, {
        "status": "trained",
        "performance_metrics": {
            "accuracy": 0.95,
            "response_time": 0.5
        }
    })
    
    # List versions
    versions = version_manager.list_versions(model_name="qa_model")
    print("\nModel versions:")
    print(json.dumps(versions, indent=2))
    
    # Example 3: Compare versions
    print("\nExample 3: Version comparison")
    # Create another version for comparison
    version_id2 = version_manager.create_version(
        model_name="qa_model",
        base_model="claude",
        config={
            "temperature": 0.5,
            "max_tokens": 150
        }
    )
    
    version_manager.update_version(version_id2, {
        "status": "trained",
        "performance_metrics": {
            "accuracy": 0.92,
            "response_time": 0.7
        }
    })
    
    comparison = version_manager.compare_versions(
        [version_id, version_id2],
        metric="accuracy"
    )
    print("\nVersion comparison:")
    print(json.dumps(comparison, indent=2))

if __name__ == "__main__":
    asyncio.run(run_training_utils_examples()) 