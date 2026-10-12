# Mixture of Experts (MoE) Implementation

This directory contains implementations of Mixture of Experts (MoE) models with both neural network-based and modality-specific approaches. The implementation includes advanced features for expert routing, load balancing, and dynamic capacity management.

## Features

### Core Features
- Neural network-based MoE with transformer architecture
- Modality-specific experts (text, vision, audio)
- Unified interface for both implementations
- Advanced expert routing and load balancing
- Dynamic expert capacity management
- Expert specialization and pruning
- Gradient checkpointing for memory efficiency

### Advanced Features
- Expert specialization through learned embeddings
- Dynamic capacity adjustment based on usage
- Expert pruning for efficiency
- Load balancing with auxiliary losses
- Noisy gating for better exploration
- Comprehensive metrics and monitoring

## Implementation Structure

```
multimind/models/moe/
├── moe_layer.py          # Base MoE layer implementation
├── moe_model.py          # Neural network-based MoE model
├── moe.py               # Modality-specific MoE implementation
├── advanced_moe.py      # Advanced MoE features
└── unified_moe.py       # Unified interface for both implementations
```

## Usage Examples

### 1. Neural MoE Example

```python
from multimind.models.moe.unified_moe import UnifiedMoE

# Create neural MoE model
model = UnifiedMoE(
    mode="neural",
    config={
        'input_dim': 768,
        'hidden_dim': 1024,
        'num_experts': 8,
        'num_layers': 6,
        'use_gradient_checkpointing': True,
        'expert_specialization': True
    }
)

# Training
result = await model.process(input_tensor, return_aux_loss=True)
output = result['output']
aux_loss = result['aux_loss']
```

### 2. Modality-based MoE Example

```python
from multimind.models.moe.unified_moe import UnifiedMoE
from multimind.models.moe.moe import TextExpert, VisionExpert

# Create modality-based MoE model
model = UnifiedMoE(
    mode="modality",
    config={'hidden_size': 768},
    experts={
        'text': TextExpert(...),
        'vision': VisionExpert(...)
    }
)

# Process multi-modal input
result = await model.process({
    'text': "Sample text",
    'vision': image_tensor
})
```

### 3. Advanced Features

```python
from multimind.models.moe.advanced_moe import AdvancedMoELayer

# Create advanced MoE layer
layer = AdvancedMoELayer(
    input_dim=768,
    num_experts=8,
    expert_dim=1024,
    use_gradient_checkpointing=True,
    expert_specialization=True,
    min_expert_capacity=4,
    max_expert_capacity=256,
    pruning_threshold=0.1
)
```

## Expert Management

### Adding Experts
```python
model.add_expert('audio', AudioExpert(...))
```

### Removing Experts
```python
model.remove_expert('audio')
```

### Getting Expert Information
```python
expert_info = model.get_expert_info()
print(f"Available experts: {expert_info['experts']}")
print(f"Modalities: {expert_info['modalities']}")
```

## Training and Fine-tuning

### Basic Training
```python
# Create trainer
trainer = MoETrainer(
    model=model,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=1000
)

# Train model
metrics = trainer.train(
    train_loader=train_loader,
    task_loss_fn=criterion,
    num_epochs=10
)
```

### Advanced Training Features
- Learning rate scheduling with warmup
- Gradient clipping
- Auxiliary losses for expert utilization
- Expert balance monitoring
- Checkpoint saving and loading

## Monitoring and Metrics

### Expert Usage Statistics
```python
stats = model.get_expert_stats()
print(f"Expert usage: {stats['usage']}")
print(f"Expert importance: {stats['importance']}")
print(f"Expert capacity: {stats['capacity']}")
```

### Performance Metrics
```python
metrics = model.get_metrics()
print(f"Training loss: {metrics['train_loss']}")
print(f"Auxiliary loss: {metrics['aux_loss']}")
```

## Best Practices

1. **Expert Configuration**
   - Start with a reasonable number of experts (8-16)
   - Use expert specialization for complex tasks
   - Enable gradient checkpointing for large models

2. **Training**
   - Use warmup for stable training
   - Monitor expert utilization
   - Adjust capacity factors based on task

3. **Memory Management**
   - Enable gradient checkpointing for large models
   - Use expert pruning for efficiency
   - Monitor memory usage during training

4. **Performance Optimization**
   - Use dynamic capacity adjustment
   - Enable expert specialization
   - Monitor and balance expert usage

## Examples

See the following example files for detailed usage:
- `moe_example.py`: Basic MoE implementation example
- `unified_example.py`: Unified interface example with both implementations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the license included in the repository. 