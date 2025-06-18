"""
Example demonstrating advanced training utilities including data augmentation, extended format support,
enhanced metrics, and visualization tools.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import yaml
import xml.etree.ElementTree as ET
from multimind.models.factory import ModelFactory
from multimind.models.multi_model import MultiModelWrapper

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

class DataFormat(Enum):
    """Extended supported training data formats."""
    JSON = "json"
    CSV = "csv"
    TXT = "txt"
    JSONL = "jsonl"
    YAML = "yaml"
    XML = "xml"
    PARQUET = "parquet"
    EXCEL = "excel"

@dataclass
class TrainingExample:
    """Extended training example with additional metadata."""
    prompt: str
    expected_response: str
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    difficulty: Optional[float] = None
    category: Optional[str] = None
    confidence: Optional[float] = None

class DataAugmentor:
    """Provides data augmentation utilities for training examples."""
    
    def __init__(self):
        self.synonym_cache = {}
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        if word in self.synonym_cache:
            return self.synonym_cache[word]
            
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name())
        
        self.synonym_cache[word] = list(set(synonyms))
        return self.synonym_cache[word]
    
    def augment_with_synonyms(
        self,
        example: TrainingExample,
        max_replacements: int = 2
    ) -> List[TrainingExample]:
        """Create variations by replacing words with synonyms."""
        augmented = []
        words = word_tokenize(example.prompt)
        
        for _ in range(max_replacements):
            new_words = words.copy()
            for i, word in enumerate(words):
                if len(word) > 3:  # Only replace words longer than 3 characters
                    synonyms = self._get_synonyms(word)
                    if synonyms:
                        new_words[i] = np.random.choice(synonyms)
                        break
            
            new_prompt = ' '.join(new_words)
            augmented.append(TrainingExample(
                prompt=new_prompt,
                expected_response=example.expected_response,
                metadata=example.metadata,
                tags=example.tags,
                difficulty=example.difficulty,
                category=example.category,
                confidence=example.confidence
            ))
        
        return augmented
    
    def augment_with_paraphrasing(
        self,
        example: TrainingExample,
        num_variations: int = 2
    ) -> List[TrainingExample]:
        """Create variations by paraphrasing the prompt."""
        # This is a simple implementation. In practice, you might want to use
        # a more sophisticated paraphrasing model or service.
        augmented = []
        words = word_tokenize(example.prompt)
        
        for _ in range(num_variations):
            # Simple word reordering
            np.random.shuffle(words)
            new_prompt = ' '.join(words)
            
            augmented.append(TrainingExample(
                prompt=new_prompt,
                expected_response=example.expected_response,
                metadata=example.metadata,
                tags=example.tags,
                difficulty=example.difficulty,
                category=example.category,
                confidence=example.confidence
            ))
        
        return augmented

class MetricsVisualizer:
    """Provides visualization tools for training metrics."""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_training_history(
        self,
        training_history: List[Dict[str, Any]],
        metrics: List[str],
        save_path: Optional[str] = None
    ):
        """Plot training history for specified metrics."""
        plt.figure(figsize=(12, 6))
        
        for metric in metrics:
            values = [record['metrics'].get(metric, 0) for record in training_history]
            epochs = [record['epoch'] for record in training_history]
            plt.plot(epochs, values, label=metric, marker='o')
        
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(self.output_dir / save_path)
        plt.close()
    
    def plot_version_comparison(
        self,
        version_metrics: Dict[str, Dict[str, float]],
        metrics: List[str],
        save_path: Optional[str] = None
    ):
        """Create bar plots comparing versions across metrics."""
        plt.figure(figsize=(15, 8))
        
        x = np.arange(len(version_metrics))
        width = 0.8 / len(metrics)
        
        for i, metric in enumerate(metrics):
            values = [data.get(metric, 0) for data in version_metrics.values()]
            plt.bar(x + i * width, values, width, label=metric)
        
        plt.xlabel('Version')
        plt.ylabel('Metric Value')
        plt.title('Version Comparison')
        plt.xticks(x + width * (len(metrics) - 1) / 2, version_metrics.keys(), rotation=45)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path)
        plt.close()
    
    def plot_confusion_matrix(
        self,
        y_true: List[str],
        y_pred: List[str],
        labels: List[str],
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix for classification results."""
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(self.output_dir / save_path)
        plt.close()
    
    def plot_metric_distribution(
        self,
        metrics: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """Plot distribution of metrics across examples."""
        plt.figure(figsize=(12, 6))
        
        for metric, values in metrics.items():
            sns.kdeplot(values, label=metric)
        
        plt.xlabel('Metric Value')
        plt.ylabel('Density')
        plt.title('Metric Distribution')
        plt.legend()
        
        if save_path:
            plt.savefig(self.output_dir / save_path)
        plt.close()

class ExtendedTrainingDataManager:
    """Extended training data manager with additional format support and augmentation."""
    
    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.examples: List[TrainingExample] = []
        self.validation_examples: List[TrainingExample] = []
        self.augmentor = DataAugmentor()
        self.visualizer = MetricsVisualizer()
    
    def load_data(self, file_path: str, format: DataFormat) -> List[TrainingExample]:
        """Load training data from extended format support."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Training data file not found: {file_path}")
        
        if format == DataFormat.YAML:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
                return [TrainingExample(**example) for example in data]
        
        elif format == DataFormat.XML:
            tree = ET.parse(file_path)
            root = tree.getroot()
            examples = []
            for example in root.findall('example'):
                examples.append(TrainingExample(
                    prompt=example.find('prompt').text,
                    expected_response=example.find('response').text,
                    metadata=self._parse_xml_metadata(example.find('metadata')),
                    tags=self._parse_xml_tags(example.find('tags'))
                ))
            return examples
        
        elif format == DataFormat.PARQUET:
            df = pd.read_parquet(file_path)
            return self._convert_dataframe_to_examples(df)
        
        elif format == DataFormat.EXCEL:
            df = pd.read_excel(file_path)
            return self._convert_dataframe_to_examples(df)
        
        else:
            # Use the original format handling
            return super().load_data(file_path, format)
    
    def _parse_xml_metadata(self, metadata_elem) -> Optional[Dict[str, Any]]:
        """Parse metadata from XML element."""
        if metadata_elem is None:
            return None
        return {child.tag: child.text for child in metadata_elem}
    
    def _parse_xml_tags(self, tags_elem) -> Optional[List[str]]:
        """Parse tags from XML element."""
        if tags_elem is None:
            return None
        return [tag.text for tag in tags_elem.findall('tag')]
    
    def _convert_dataframe_to_examples(self, df: pd.DataFrame) -> List[TrainingExample]:
        """Convert DataFrame to training examples."""
        return [
            TrainingExample(
                prompt=row['prompt'],
                expected_response=row['expected_response'],
                metadata=row.get('metadata'),
                tags=row.get('tags', '').split(',') if 'tags' in row else None,
                difficulty=row.get('difficulty'),
                category=row.get('category'),
                confidence=row.get('confidence')
            )
            for _, row in df.iterrows()
        ]
    
    def augment_data(
        self,
        examples: List[TrainingExample],
        augmentation_methods: List[str] = ['synonyms', 'paraphrasing'],
        **kwargs
    ) -> List[TrainingExample]:
        """Augment training data using specified methods."""
        augmented = []
        for example in examples:
            augmented.append(example)  # Keep original example
            
            if 'synonyms' in augmentation_methods:
                augmented.extend(
                    self.augmentor.augment_with_synonyms(
                        example,
                        max_replacements=kwargs.get('max_replacements', 2)
                    )
                )
            
            if 'paraphrasing' in augmentation_methods:
                augmented.extend(
                    self.augmentor.augment_with_paraphrasing(
                        example,
                        num_variations=kwargs.get('num_variations', 2)
                    )
                )
        
        return augmented
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze training data quality with extended metrics."""
        stats = self.validate_examples()
        
        # Add additional metrics
        if self.examples:
            # Calculate difficulty distribution
            difficulties = [ex.difficulty for ex in self.examples if ex.difficulty is not None]
            if difficulties:
                stats['difficulty_stats'] = {
                    'mean': np.mean(difficulties),
                    'std': np.std(difficulties),
                    'min': np.min(difficulties),
                    'max': np.max(difficulties)
                }
            
            # Calculate category distribution
            categories = [ex.category for ex in self.examples if ex.category is not None]
            if categories:
                stats['category_distribution'] = {
                    category: categories.count(category)
                    for category in set(categories)
                }
            
            # Calculate confidence distribution
            confidences = [ex.confidence for ex in self.examples if ex.confidence is not None]
            if confidences:
                stats['confidence_stats'] = {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences)
                }
        
        return stats
    
    def visualize_data_quality(self, save_dir: Optional[str] = None):
        """Create visualizations for data quality metrics."""
        stats = self.analyze_data_quality()
        
        # Plot difficulty distribution
        if 'difficulty_stats' in stats:
            difficulties = [ex.difficulty for ex in self.examples if ex.difficulty is not None]
            self.visualizer.plot_metric_distribution(
                {'difficulty': difficulties},
                save_path='difficulty_distribution.png' if save_dir else None
            )
        
        # Plot category distribution
        if 'category_distribution' in stats:
            categories = [ex.category for ex in self.examples if ex.category is not None]
            plt.figure(figsize=(10, 6))
            sns.countplot(y=categories)
            plt.title('Category Distribution')
            if save_dir:
                plt.savefig(Path(save_dir) / 'category_distribution.png')
            plt.close()
        
        # Plot confidence distribution
        if 'confidence_stats' in stats:
            confidences = [ex.confidence for ex in self.examples if ex.confidence is not None]
            self.visualizer.plot_metric_distribution(
                {'confidence': confidences},
                save_path='confidence_distribution.png' if save_dir else None
            )

async def run_advanced_training_utils_examples():
    # Initialize the extended training data manager
    data_manager = ExtendedTrainingDataManager()
    
    # Example 1: Load and prepare data with extended formats
    print("Example 1: Loading data with extended formats")
    examples = [
        TrainingExample(
            prompt="What is the capital of France?",
            expected_response="Paris",
            tags=["geography", "capital"],
            difficulty=0.3,
            category="geography",
            confidence=0.95
        ),
        TrainingExample(
            prompt="Who wrote Romeo and Juliet?",
            expected_response="William Shakespeare",
            tags=["literature", "author"],
            difficulty=0.5,
            category="literature",
            confidence=0.98
        )
    ]
    
    # Example 2: Data augmentation
    print("\nExample 2: Data augmentation")
    augmented_examples = data_manager.augment_data(
        examples,
        augmentation_methods=['synonyms', 'paraphrasing'],
        max_replacements=2,
        num_variations=2
    )
    print(f"Original examples: {len(examples)}")
    print(f"Augmented examples: {len(augmented_examples)}")
    
    # Example 3: Data quality analysis
    print("\nExample 3: Data quality analysis")
    quality_stats = data_manager.analyze_data_quality()
    print("\nData quality stats:")
    print(json.dumps(quality_stats, indent=2))
    
    # Example 4: Data visualization
    print("\nExample 4: Data visualization")
    data_manager.visualize_data_quality(save_dir="visualizations")
    
    # Example 5: Training history visualization
    print("\nExample 5: Training history visualization")
    training_history = [
        {'epoch': 0, 'metrics': {'accuracy': 0.8, 'loss': 0.2}},
        {'epoch': 1, 'metrics': {'accuracy': 0.85, 'loss': 0.15}},
        {'epoch': 2, 'metrics': {'accuracy': 0.9, 'loss': 0.1}}
    ]
    
    data_manager.visualizer.plot_training_history(
        training_history,
        metrics=['accuracy', 'loss'],
        save_path='training_history.png'
    )
    
    # Example 6: Version comparison visualization
    print("\nExample 6: Version comparison visualization")
    version_metrics = {
        'v1': {'accuracy': 0.95, 'response_time': 0.5},
        'v2': {'accuracy': 0.92, 'response_time': 0.7}
    }
    
    data_manager.visualizer.plot_version_comparison(
        version_metrics,
        metrics=['accuracy', 'response_time'],
        save_path='version_comparison.png'
    )

if __name__ == "__main__":
    asyncio.run(run_advanced_training_utils_examples()) 