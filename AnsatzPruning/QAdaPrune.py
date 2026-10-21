"""
QAdaPrune: Quantum Adaptive Pruning
Tracks parameter importance and prunes low-impact parameters during VQE optimization.

Based on the QAdaPrune paper, integrated with Momentum Builder and Monte Carlo optimization.
Paper: https://arxiv.org/abs/2408.13352
Github link: https://github.com/aicaffeinelife/QAdaPrune
"""

import numpy as np
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from scipy.optimize import minimize


@dataclass
class ParameterMetrics:
    """Track metrics for a single parameter"""
    param_id: int
    qubit_idx: int
    gate_type: str  # 'rx', 'ry', 'rz'
    layer: int
    
    # Importance tracking
    gradient: float = 0.0
    importance_score: float = 0.0  # Γ(θ)
    avg_gradient: float = 0.0
    gradient_variance: float = 0.0
    
    # State tracking
    is_pruned: bool = False
    pruning_iteration: int = -1
    frozen_value: Optional[float] = None
    
    # Update history
    update_history: List[float] = None
    gradient_history: List[float] = None
    
    def __post_init__(self):
        if self.update_history is None:
            self.update_history = []
        if self.gradient_history is None:
            self.gradient_history = []


class QAdaPruner:
    """
    QAdaPrune algorithm implementation.
    Tracks parameter importance and prunes parameters with low impact on cost function.
    """
    
    def __init__(self, num_qubits: int, pruning_threshold: float = 0.01, 
                 window_size: int = 5, patience: int = 3):
        """
        Args:
            num_qubits: Number of qubits in circuit
            pruning_threshold: Threshold for pruning (relative to max importance)
            window_size: Number of iterations to track for moving average
            patience: Iterations to wait before declaring convergence
        """
        self.num_qubits = num_qubits
        self.pruning_threshold = pruning_threshold
        self.window_size = window_size
        self.patience = patience
        
        self.parameters: Dict[int, ParameterMetrics] = {}
        self.iteration = 0
        self.best_cost = float('inf')
        self.cost_history = []
        self.convergence_counter = 0
        self.pruned_count = 0
        
    def register_parameter(self, param_id: int, qubit_idx: int, 
                          gate_type: str, layer: int):
        """Register a new parameter for tracking"""
        self.parameters[param_id] = ParameterMetrics(
            param_id=param_id,
            qubit_idx=qubit_idx,
            gate_type=gate_type,
            layer=layer
        )
    
    def update_importance_scores(self, param_gradients: np.ndarray):
        """
        Update importance scores based on parameter gradients.
        
        Importance Γ(θ) is computed as the exponential moving average of |∇θ|
        """
        for param_id, grad_val in enumerate(param_gradients):
            if param_id in self.parameters and not self.parameters[param_id].is_pruned:
                metrics = self.parameters[param_id]
                
                # Update gradient history
                metrics.gradient = abs(grad_val)
                metrics.gradient_history.append(metrics.gradient)
                
                # Keep only recent history (window)
                if len(metrics.gradient_history) > self.window_size:
                    metrics.gradient_history.pop(0)
                
                # Compute importance as moving average of gradient magnitude
                metrics.avg_gradient = np.mean(metrics.gradient_history)
                
                # Compute variance for stability assessment
                if len(metrics.gradient_history) > 1:
                    metrics.gradient_variance = np.var(metrics.gradient_history)
                
                # Importance score: weighted by recency
                decay_weights = np.array([0.5 ** (len(metrics.gradient_history) - 1 - i) 
                                         for i in range(len(metrics.gradient_history))])
                decay_weights /= decay_weights.sum()
                metrics.importance_score = np.sum(
                    decay_weights * np.array(metrics.gradient_history)
                )
    
    def get_pruning_candidates(self) -> List[int]:
        """
        Identify parameters that should be pruned.
        
        A parameter is pruned if its importance score is below the threshold
        relative to the maximum importance in its layer.
        """
        if not self.parameters:
            return []
        
        # Group by layer
        layers = {}
        for param_id, metrics in self.parameters.items():
            if not metrics.is_pruned:
                if metrics.layer not in layers:
                    layers[metrics.layer] = []
                layers[metrics.layer].append((param_id, metrics.importance_score))
        
        candidates = []
        for layer, params_in_layer in layers.items():
            if not params_in_layer:
                continue
                
            # Find max importance in this layer
            max_importance = max(score for _, score in params_in_layer)
            
            # Prune if below threshold
            threshold = self.pruning_threshold * max_importance
            for param_id, score in params_in_layer:
                if score < threshold:
                    candidates.append(param_id)
        
        return candidates
    
    def prune_parameters(self, candidates: List[int], 
                        param_values: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Prune identified low-importance parameters.
        
        Returns:
            - Updated parameter values (with pruned params removed)
            - List of pruned parameter IDs
        """
        if not candidates:
            return param_values, []
        
        pruned = []
        for param_id in candidates:
            if param_id in self.parameters:
                metrics = self.parameters[param_id]
                if not metrics.is_pruned:
                    metrics.is_pruned = True
                    metrics.pruning_iteration = self.iteration
                    metrics.frozen_value = param_values[param_id] if param_id < len(param_values) else 0.0
                    pruned.append(param_id)
                    self.pruned_count += 1
        
        # Create mapping of old param indices to new indices
        active_indices = [i for i in range(len(param_values)) 
                         if i not in pruned]
        new_param_values = param_values[active_indices]
        
        return new_param_values, pruned
    
    def update_iteration(self, cost_value: float):
        """Update iteration counter and track cost evolution"""
        self.iteration += 1
        self.cost_history.append(cost_value)
        
        # Check convergence
        if cost_value < self.best_cost:
            self.best_cost = cost_value
            self.convergence_counter = 0
        else:
            self.convergence_counter += 1
    
    def should_add_layer(self) -> bool:
        """Determine if we should add another layer"""
        if len(self.cost_history) < 2:
            return True
        
        # Check for cost improvement
        recent_improvement = abs(
            self.cost_history[-1] - self.cost_history[-2]
        ) / (abs(self.cost_history[-2]) + 1e-10)
        
        # Add layer if still improving or within patience
        return recent_improvement > 1e-4 or self.convergence_counter < self.patience
    
    def has_converged(self) -> bool:
        """Check if optimization has converged"""
        return self.convergence_counter >= self.patience
    
    def get_pruning_report(self) -> Dict:
        """Generate a report of pruning statistics"""
        active_params = [m for m in self.parameters.values() if not m.is_pruned]
        pruned_params = [m for m in self.parameters.values() if m.is_pruned]
        
        return {
            'iteration': self.iteration,
            'total_parameters': len(self.parameters),
            'active_parameters': len(active_params),
            'pruned_parameters': len(pruned_params),
            'pruning_rate': len(pruned_params) / max(len(self.parameters), 1),
            'best_cost': self.best_cost,
            'current_cost': self.cost_history[-1] if self.cost_history else None,
            'convergence_counter': self.convergence_counter,
        }


class QAdaPrunedCircuit:
    """
    Manages circuit construction with QAdaPrune integration.
    Tracks which parameters are active vs pruned.
    """
    
    def __init__(self, base_circuit: QuantumCircuit, pruner: QAdaPruner):
        self.base_circuit = base_circuit
        self.pruner = pruner
        self.full_circuit = base_circuit.copy()
        self.param_mapping = {}  # Maps param_id to (qubit, gate_type)
        self.active_param_ids = []
        
    def add_momentum_layer(self, layer_idx: int, active_qubits: List[int], 
                          gate_type: str = 'rx') -> QuantumCircuit:
        """
        Add a new momentum layer to the circuit.
        
        Args:
            layer_idx: Which layer this is
            active_qubits: Qubits to apply gates to
            gate_type: Type of rotation gate ('rx', 'ry', 'rz')
        
        Returns:
            Updated circuit with new layer
        """
        n_qubits = self.full_circuit.num_qubits
        new_layer = QuantumCircuit(n_qubits)
        
        # Create parameters for this layer
        layer_params = []
        for qubit_idx in active_qubits:
            param_id = len(self.active_param_ids)
            param = Parameter(f"θ_{layer_idx}_{qubit_idx}")
            layer_params.append(param)
            
            # Register with pruner
            self.pruner.register_parameter(param_id, qubit_idx, gate_type, layer_idx)
            self.active_param_ids.append(param_id)
            self.param_mapping[param_id] = (qubit_idx, gate_type)
            
            # Add gate to layer
            if gate_type == 'rx':
                new_layer.rx(param, qubit_idx)
            elif gate_type == 'ry':
                new_layer.ry(param, qubit_idx)
            elif gate_type == 'rz':
                new_layer.rz(param, qubit_idx)
        
        # Add entangling gates
        for i, q in enumerate(active_qubits[:-1]):
            new_layer.cx(q, active_qubits[i + 1])
        
        self.full_circuit = self.full_circuit.compose(new_layer)
        return self.full_circuit
    
    def remove_pruned_gates(self, pruned_param_ids: List[int]):
        """
        Remove pruned gates from the circuit (conceptually freeze their values).
        In practice, we track which gates are frozen rather than modifying the circuit.
        """
        for param_id in pruned_param_ids:
            if param_id in self.active_param_ids:
                self.active_param_ids.remove(param_id)
    
    def get_num_active_parameters(self) -> int:
        """Return number of currently active (non-pruned) parameters"""
        return len(self.active_param_ids)


def calculate_parameter_gradients(params: np.ndarray, 
                                 circuit: QuantumCircuit,
                                 hamiltonian: SparsePauliOp,
                                 estimator: Estimator,
                                 delta: float = np.pi / 2) -> np.ndarray:
    """
    Calculate gradients for all parameters using parameter shift rule.
    
    Args:
        params: Current parameter values
        circuit: Quantum circuit
        hamiltonian: Hamiltonian to minimize
        estimator: Estimator for expectation values
        delta: Shift amount for parameter-shift rule
    
    Returns:
        Gradient vector
    """
    gradients = np.zeros_like(params)
    
    for i in range(len(params)):
        # Forward shift
        params_plus = params.copy()
        params_plus[i] += delta
        
        # Backward shift
        params_minus = params.copy()
        params_minus[i] -= delta
        
        # Compute expectations
        pub_plus = (circuit, hamiltonian, params_plus)
        pub_minus = (circuit, hamiltonian, params_minus)
        
        try:
            cost_plus = estimator.run([pub_plus]).result()[0].data.evs[-1]
            cost_minus = estimator.run([pub_minus]).result()[0].data.evs[-1]
            gradients[i] = (cost_plus - cost_minus) / (2 * delta)
        except:
            gradients[i] = 0.0
    
    return gradients