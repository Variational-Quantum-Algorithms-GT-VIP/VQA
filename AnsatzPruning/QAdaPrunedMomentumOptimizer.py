"""
QAdaPruned Momentum Optimizer
Combines QAdaPrune parameter pruning with Momentum-based layer addition and Monte Carlo optimization.
"""

import numpy as np
import heapq
import time
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from scipy.optimize import minimize, differential_evolution
from qiskit_aer.aerprovider import AerSimulator

from QAdaPrune import QAdaPruner, QAdaPrunedCircuit, calculate_parameter_gradients


@dataclass
class OptimizationConfig:
    """Configuration for QAdaPrune optimization"""
    max_layers: int = 5
    pruning_threshold: float = 0.01
    momentum_beta1: float = 0.9
    momentum_beta2: float = 0.99
    pruning_window_size: int = 5
    patience: int = 3
    optimization_method: str = "cobyla"  # 'cobyla', 'de', 'sa'
    max_iterations_per_layer: int = 100
    de_popsize: int = 15


class MomentumTracker:
    """Tracks momentum for gradient-based layer expansion"""
    
    def __init__(self, num_qubits: int, beta1: float = 0.9):
        self.num_qubits = num_qubits
        self.beta1 = beta1
        self.momentum = np.zeros(num_qubits)
        self.iteration = 0
    
    def update(self, gradients: np.ndarray) -> np.ndarray:
        """Update momentum based on gradients"""
        qubit_gradients = np.zeros(self.num_qubits)
        
        # Aggregate gradients per qubit
        if len(gradients) > 0:
            for i, grad in enumerate(gradients):
                qubit_idx = i % self.num_qubits
                qubit_gradients[qubit_idx] += abs(grad)
        
        # Update momentum with exponential moving average
        self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * qubit_gradients
        self.iteration += 1
        
        return self.momentum
    
    def get_most_active_qubits(self, k: int) -> List[int]:
        """Get indices of k qubits with highest momentum"""
        if k >= self.num_qubits:
            return list(range(self.num_qubits))
        
        indices = np.argsort(-self.momentum)[:k]
        return sorted(indices.tolist())


class QAdaPrunedMomentumOptimizer:
    """
    Main optimizer combining QAdaPrune with Momentum-based layer addition.
    """
    
    def __init__(self, circuit: QuantumCircuit, hamiltonian: SparsePauliOp,
                 estimator: Estimator, config: OptimizationConfig = None):
        """
        Args:
            circuit: Base quantum circuit (usually just qubits)
            hamiltonian: Hamiltonian to minimize (SparsePauliOp)
            estimator: Qiskit StatevectorEstimator for cost evaluation
            config: Optimization configuration
        """
        self.base_circuit = circuit
        self.hamiltonian = hamiltonian
        self.estimator = estimator
        self.config = config or OptimizationConfig()
        
        self.num_qubits = circuit.num_qubits
        self.observables = [*hamiltonian.paulis, hamiltonian]
        
        # Initialize pruner and momentum tracker
        self.pruner = QAdaPruner(
            num_qubits=self.num_qubits,
            pruning_threshold=self.config.pruning_threshold,
            window_size=self.config.pruning_window_size,
            patience=self.config.patience
        )
        
        self.momentum_tracker = MomentumTracker(
            num_qubits=self.num_qubits,
            beta1=self.config.momentum_beta1
        )
        
        # Circuit tracking
        self.pruned_circuit = QAdaPrunedCircuit(circuit, self.pruner)
        self.current_params = np.array([])
        self.layer_count = 0
        
        # Statistics
        self.history = {
            'costs': [],
            'param_counts': [],
            'pruned_counts': [],
            'layer_sizes': [],
            'layer_times': [],
            'gradients': []
        }
    
    def _cost_function(self, params: np.ndarray) -> float:
        if len(params) == 0:
            return 0.0
        pub = (self.pruned_circuit.full_circuit, self.hamiltonian, params)
        cost = self.estimator.run([pub]).result()[0].data.evs
        return float(cost)
    
    def _optimize_layer_params(self, initial_params: np.ndarray,
                               max_iterations: int = None) -> np.ndarray:
        """Optimize parameters for current layer"""
        
        if len(initial_params) == 0:
            return initial_params
        
        max_iters = max_iterations or self.config.max_iterations_per_layer
        
        if self.config.optimization_method == "cobyla":
            result = minimize(
                self._cost_function,
                initial_params,
                method="COBYLA",
                options={'maxiter': max_iters, 'tol': 1e-6}
            )
            return result.x
        
        elif self.config.optimization_method == "de":
            bounds = [(-2*np.pi, 2*np.pi) for _ in initial_params]
            result = differential_evolution(
                self._cost_function,
                bounds,
                seed=42,
                maxiter=max_iters,
                popsize=self.config.de_popsize
            )
            return result.x
        
        else:
            # Default to COBYLA
            result = minimize(
                self._cost_function,
                initial_params,
                method="COBYLA",
                options={'maxiter': max_iters}
            )
            return result.x
    
    def add_and_optimize_layer(self) -> Tuple[float, int]:
        """
        Add a new momentum-guided layer and optimize its parameters.
        
        Returns:
            - Cost after optimization
            - Number of parameters in new layer
        """
        layer_start_time = time.time()
        
        # Get active qubits based on momentum
        if self.layer_count == 0:
            # First layer: use all qubits
            active_qubits = list(range(self.num_qubits))
        else:
            # Subsequent layers: select based on momentum
            k = max(2, self.num_qubits // 2)  # Use at least 2 qubits
            active_qubits = self.momentum_tracker.get_most_active_qubits(k)
        
        # Add layer to circuit
        self.pruned_circuit.add_momentum_layer(
            layer_idx=self.layer_count,
            active_qubits=active_qubits,
            gate_type='rx'
        )
        
        # Initialize new parameters
        new_param_count = len(active_qubits)
        new_params = np.ones(new_param_count) * np.pi / 4
        self.current_params = np.concatenate([self.current_params, new_params])
        
        # Optimize this layer's parameters
        self.current_params = self._optimize_layer_params(self.current_params)
        
        # Calculate cost and gradients
        cost = self._cost_function(self.current_params)
        gradients = calculate_parameter_gradients(
            self.current_params,
            self.pruned_circuit.full_circuit,
            self.hamiltonian,
            self.estimator
        )
        
        # Update pruner with gradients
        self.pruner.update_importance_scores(gradients)
        self.momentum_tracker.update(gradients[:self.num_qubits])
        
        # Check for pruning candidates
        candidates = self.pruner.get_pruning_candidates()
        
        if candidates:
            print(f"  → Pruning {len(candidates)} parameters")
            self.current_params, pruned = self.pruner.prune_parameters(
                candidates, self.current_params
            )
            self.pruned_circuit.remove_pruned_gates(pruned)
        
        # Update iteration and history
        self.pruner.update_iteration(cost)
        
        layer_time = time.time() - layer_start_time
        self.history['costs'].append(cost)
        self.history['param_counts'].append(len(self.current_params))
        self.history['pruned_counts'].append(self.pruner.pruned_count)
        self.history['layer_sizes'].append(new_param_count)
        self.history['layer_times'].append(layer_time)
        self.history['gradients'].append(np.max(np.abs(gradients)))
        
        self.layer_count += 1
        
        return cost, new_param_count
    
    def optimize(self, verbose: bool = True) -> Dict:
        """
        Run the full QAdaPrune + Momentum optimization pipeline.
        
        Returns:
            Dictionary with optimization results
        """
        print("=" * 60)
        print("QAdaPrune + Momentum Optimizer")
        print("=" * 60)
        
        total_start_time = time.time()
        
        for layer_idx in range(self.config.max_layers):
            if verbose:
                print(f"\n[Layer {layer_idx}]")
                print(f"  Current parameters: {len(self.current_params)}")
                print(f"  Total pruned: {self.pruner.pruned_count}")
            
            # Add and optimize layer
            cost, layer_params = self.add_and_optimize_layer()
            
            if verbose:
                report = self.pruner.get_pruning_report()
                print(f"  Cost: {cost:.6f}")
                print(f"  Active params: {report['active_parameters']}")
                print(f"  Pruning rate: {report['pruning_rate']:.1%}")
                print(f"  Time: {self.history['layer_times'][-1]:.3f}s")
            
            # Check convergence
            if self.pruner.has_converged():
                if verbose:
                    print(f"\n✓ Converged after {layer_idx + 1} layers")
                break
        
        total_time = time.time() - total_start_time
        
        # Prepare results
        results = {
            'optimized_circuit': self.pruned_circuit.full_circuit,
            'optimized_params': self.current_params,
            'final_cost': self.history['costs'][-1] if self.history['costs'] else None,
            'num_layers': self.layer_count,
            'num_parameters': len(self.current_params),
            'pruned_parameters': self.pruner.pruned_count,
            'total_time': total_time,
            'history': self.history,
            'pruning_report': self.pruner.get_pruning_report()
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print("Optimization Complete")
            print("=" * 60)
            print(f"Final Cost: {results['final_cost']:.6f}")
            print(f"Layers Added: {self.layer_count}")
            print(f"Final Parameters: {results['num_parameters']}")
            print(f"Parameters Pruned: {results['pruned_parameters']}")
            print(f"Total Time: {total_time:.2f}s")
            print("=" * 60)
        
        return results


def compare_ansatze(base_circuit: QuantumCircuit,
                    hamiltonian: SparsePauliOp,
                    estimator: Estimator,
                    num_runs: int = 5) -> Dict:
    """
    Compare QAdaPrune approach vs standard approaches.
    
    Returns statistics for benchmarking
    """
    results = {
        'qadaprune': [],
        'momentum_only': [],
        'efficient_su2': []
    }
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # QAdaPrune + Momentum
        config = OptimizationConfig(
            max_layers=5,
            pruning_threshold=0.01,
            optimization_method='cobyla'
        )
        optimizer = QAdaPrunedMomentumOptimizer(
            base_circuit.copy(),
            hamiltonian,
            estimator,
            config
        )
        qadaprune_results = optimizer.optimize(verbose=False)
        results['qadaprune'].append(qadaprune_results)
    
    return results