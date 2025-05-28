"""
Neuromorphic Spiking Memory implementation using LIF neurons and STDP learning.
"""

from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
from .base import BaseMemory
from .vector_store import VectorStoreMemory

class LIFNeuron:
    """Leaky Integrate-and-Fire neuron implementation."""
    def __init__(
        self,
        threshold: float = 1.0,
        decay_rate: float = 0.1,
        refractory_period: float = 0.1
    ):
        self.threshold = threshold
        self.decay_rate = decay_rate
        self.refractory_period = refractory_period
        self.membrane_potential = 0.0
        self.last_spike_time = -np.inf
        self.spike_history = []

    def update(self, input_current: float, current_time: float) -> bool:
        """Update neuron state and return whether it spiked."""
        # Check refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            return False

        # Update membrane potential
        self.membrane_potential = (
            self.membrane_potential * np.exp(-self.decay_rate) +
            input_current
        )

        # Check for spike
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = 0.0
            self.last_spike_time = current_time
            self.spike_history.append(current_time)
            return True

        return False

class STDP:
    """Spike-Timing-Dependent Plasticity implementation."""
    def __init__(
        self,
        learning_rate: float = 0.01,
        tau_plus: float = 0.02,
        tau_minus: float = 0.02,
        weight_max: float = 1.0
    ):
        self.learning_rate = learning_rate
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.weight_max = weight_max
        self.weights = defaultdict(lambda: 0.5)

    def update(
        self,
        pre_spike_time: float,
        post_spike_time: float,
        pre_id: str,
        post_id: str
    ) -> None:
        """Update synaptic weights based on spike timing."""
        dt = post_spike_time - pre_spike_time
        
        if dt > 0:  # Pre-before-post: LTP
            dw = self.learning_rate * np.exp(-dt / self.tau_plus)
            self.weights[(pre_id, post_id)] = min(
                self.weights[(pre_id, post_id)] + dw,
                self.weight_max
            )
        else:  # Post-before-pre: LTD
            dw = -self.learning_rate * np.exp(dt / self.tau_minus)
            self.weights[(pre_id, post_id)] = max(
                self.weights[(pre_id, post_id)] + dw,
                0.0
            )

class SpikingMemory(BaseMemory):
    """Memory implementation using neuromorphic spiking networks."""

    def __init__(
        self,
        neuron_threshold: float = 1.0,
        neuron_decay: float = 0.1,
        refractory_period: float = 0.1,
        stdp_learning_rate: float = 0.01,
        stdp_tau_plus: float = 0.02,
        stdp_tau_minus: float = 0.02,
        max_neurons: int = 1000,
        **kwargs
    ):
        """Initialize spiking memory."""
        super().__init__(**kwargs)
        
        # Neuron parameters
        self.neuron_threshold = neuron_threshold
        self.neuron_decay = neuron_decay
        self.refractory_period = refractory_period
        self.max_neurons = max_neurons
        
        # Component memories
        self.vector_memory = VectorStoreMemory()
        
        # Neural network components
        self.neurons: Dict[str, LIFNeuron] = {}
        self.stdp = STDP(
            learning_rate=stdp_learning_rate,
            tau_plus=stdp_tau_plus,
            tau_minus=stdp_tau_minus
        )
        
        # Memory tracking
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.neuron_mappings: Dict[str, str] = {}  # memory_id -> neuron_id
        self.spike_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Statistics
        self.total_spikes = 0
        self.total_neurons = 0
        self.avg_firing_rate = 0.0

    async def add_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new memory with spiking representation."""
        # Create memory entry
        memory = {
            'id': memory_id,
            'content': content,
            'created_at': datetime.now(),
            'last_accessed': datetime.now(),
            'access_count': 0,
            'metadata': metadata or {}
        }
        
        # Store memory
        self.memories[memory_id] = memory
        
        # Create neuron if under limit
        if self.total_neurons < self.max_neurons:
            neuron_id = f"neuron_{self.total_neurons}"
            self.neurons[neuron_id] = LIFNeuron(
                threshold=self.neuron_threshold,
                decay_rate=self.neuron_decay,
                refractory_period=self.refractory_period
            )
            self.neuron_mappings[memory_id] = neuron_id
            self.total_neurons += 1
        
        # Add to vector memory
        await self.vector_memory.add(memory_id, content, metadata)

    async def get_memory(
        self,
        memory_id: str,
        current_time: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Get a memory by ID, updating neural activity."""
        if memory_id not in self.memories:
            return None
            
        memory = self.memories[memory_id]
        
        # Update access tracking
        memory['access_count'] += 1
        memory['last_accessed'] = datetime.now()
        
        # Update neural activity
        if current_time is not None and memory_id in self.neuron_mappings:
            neuron_id = self.neuron_mappings[memory_id]
            neuron = self.neurons[neuron_id]
            
            # Simulate input current based on memory access
            input_current = 1.0  # Placeholder for actual input calculation
            
            # Update neuron and record spike
            if neuron.update(input_current, current_time):
                self.total_spikes += 1
                self.spike_patterns[memory_id].append(current_time)
                
                # Update STDP for connected neurons
                for other_id, other_neuron in self.neurons.items():
                    if other_id != neuron_id and other_neuron.spike_history:
                        self.stdp.update(
                            other_neuron.spike_history[-1],
                            current_time,
                            other_id,
                            neuron_id
                        )
        
        return memory

    async def get_spike_history(
        self,
        memory_id: str,
        time_window: Optional[float] = None
    ) -> List[float]:
        """Get spike history for a memory."""
        if memory_id not in self.spike_patterns:
            return []
            
        spikes = self.spike_patterns[memory_id]
        if time_window:
            current_time = spikes[-1] if spikes else 0.0
            spikes = [t for t in spikes if current_time - t <= time_window]
        return spikes

    async def get_neuron_stats(
        self,
        neuron_id: str
    ) -> Dict[str, Any]:
        """Get statistics for a neuron."""
        if neuron_id not in self.neurons:
            return {}
            
        neuron = self.neurons[neuron_id]
        spikes = neuron.spike_history
        
        if not spikes:
            return {
                'total_spikes': 0,
                'firing_rate': 0.0,
                'last_spike': None
            }
            
        return {
            'total_spikes': len(spikes),
            'firing_rate': len(spikes) / (spikes[-1] - spikes[0]) if len(spikes) > 1 else 0.0,
            'last_spike': spikes[-1]
        }

    async def get_synaptic_weights(
        self,
        neuron_id: str
    ) -> Dict[Tuple[str, str], float]:
        """Get synaptic weights for a neuron."""
        return {
            (pre, post): weight
            for (pre, post), weight in self.stdp.weights.items()
            if pre == neuron_id or post == neuron_id
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        # Calculate average firing rate
        firing_rates = []
        for neuron in self.neurons.values():
            if neuron.spike_history:
                spikes = neuron.spike_history
                if len(spikes) > 1:
                    firing_rates.append(len(spikes) / (spikes[-1] - spikes[0]))
        
        self.avg_firing_rate = np.mean(firing_rates) if firing_rates else 0.0
        
        return {
            'total_memories': len(self.memories),
            'total_neurons': self.total_neurons,
            'total_spikes': self.total_spikes,
            'avg_firing_rate': self.avg_firing_rate,
            'active_neurons': sum(
                1 for neuron in self.neurons.values()
                if neuron.spike_history
            )
        }

    async def remove_memory(self, memory_id: str) -> None:
        """Remove a memory and its neural representation."""
        if memory_id in self.memories:
            # Remove from vector memory
            await self.vector_memory.remove(memory_id)
            
            # Remove neural representation
            if memory_id in self.neuron_mappings:
                neuron_id = self.neuron_mappings[memory_id]
                del self.neurons[neuron_id]
                del self.neuron_mappings[memory_id]
                self.total_neurons -= 1
            
            # Remove from tracking
            del self.memories[memory_id]
            if memory_id in self.spike_patterns:
                del self.spike_patterns[memory_id] 