# UnifiedQAdaPruneOptimizer.py
"""
Unified QAdaPrune Optimizer for all problem types
Single method call for any Hamiltonian
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator

from QAdaPrunedMomentumOptimizer import QAdaPrunedMomentumOptimizer, OptimizationConfig


@dataclass
class OptimizationResult:
    """Unified result format for all problems"""
    problem_name: str
    num_qubits: int
    hamiltonian: SparsePauliOp
    final_cost: float
    num_layers: int
    num_parameters: int
    pruned_parameters: int
    total_time: float
    history: Dict
    optimization_report: Dict
    
    def __str__(self):
        """Pretty print results"""
        return f"""
╔══════════════════════════════════════════════════════════╗
║  {self.problem_name:^52} ║
╚══════════════════════════════════════════════════════════╝

Final Cost:              {self.final_cost:.6f}
Number of Layers:       {self.num_layers}
Active Parameters:      {self.num_parameters}
Pruned Parameters:      {self.pruned_parameters}
Total Optimization Time: {self.total_time:.3f}s

Hamiltonian:
  Num Qubits:          {self.num_qubits}
  Num Terms:           {len(self.hamiltonian.paulis)}

Optimization History:
"""


class UnifiedQAdaPruneOptimizer:
    """
    Main optimizer class - unified interface for all problems
    
    Usage:
        optimizer = UnifiedQAdaPruneOptimizer()
        result = optimizer.optimize(problem_name, hamiltonian, num_qubits)
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize optimizer with configuration
        
        Args:
            config: OptimizationConfig object (uses defaults if None)
        """
        self.config = config or OptimizationConfig(
            max_layers=4,
            pruning_threshold=0.01,
            momentum_beta1=0.9,
            patience=3
        )
        self.estimator = Estimator()
        self.results_history = []
    
    def optimize(self,
                 problem_name: str,
                 hamiltonian: SparsePauliOp,
                 num_qubits: int,
                 verbose: bool = True) -> OptimizationResult:
        """
        Optimize for any Hamiltonian (Main Method - Use This!)
        
        Args:
            problem_name: Name of problem (e.g., 'KNAPSACK', 'TSP', 'SET_COVER')
            hamiltonian: SparsePauliOp representing the problem
            num_qubits: Number of qubits
            verbose: Print progress
        
        Returns:
            OptimizationResult with all data
        
        Example:
            >>> optimizer = UnifiedQAdaPruneOptimizer()
            >>> H = SparsePauliOp.from_list([("ZIZZ", 7), ("ZZII", 1)])
            >>> result = optimizer.optimize("MY_PROBLEM", H, 4)
            >>> print(result)
            >>> print(result.final_cost)
        """
        
        if verbose:
            print("\n" + "="*60)
            print(f"Optimizing: {problem_name}")
            print("="*60)
            print(f"Qubits: {num_qubits}")
            print(f"Hamiltonian terms: {len(hamiltonian.paulis)}")
        
        # Validate inputs
        assert num_qubits > 0, "num_qubits must be > 0"
        assert hamiltonian.num_qubits == num_qubits, \
            f"Hamiltonian qubits ({hamiltonian.num_qubits}) != num_qubits ({num_qubits})"
        
        # Create circuit
        circuit = QuantumCircuit(num_qubits)
        
        # Run optimization
        start_time = time.time()
        
        try:
            optimizer = QAdaPrunedMomentumOptimizer(
                circuit=circuit,
                hamiltonian=hamiltonian,
                estimator=self.estimator,
                config=self.config
            )
            
            opt_results = optimizer.optimize(verbose=verbose)
            
            elapsed = time.time() - start_time
            
            # Package results
            result = OptimizationResult(
                problem_name=problem_name,
                num_qubits=num_qubits,
                hamiltonian=hamiltonian,
                final_cost=opt_results['final_cost'],
                num_layers=opt_results['num_layers'],
                num_parameters=opt_results['num_parameters'],
                pruned_parameters=opt_results['pruned_parameters'],
                total_time=elapsed,
                history=opt_results['history'],
                optimization_report=opt_results['pruning_report']
            )
            
            # Store in history
            self.results_history.append(result)
            
            if verbose:
                print(result)
            
            return result
            
        except Exception as e:
            if verbose:
                print(f"\n✗ Optimization failed: {e}")
                import traceback
                traceback.print_exc()
            raise
    
    def optimize_multiple(self,
                         problems: Dict[str, Tuple[SparsePauliOp, int]],
                         verbose: bool = True) -> Dict[str, OptimizationResult]:
        """
        Optimize multiple problems in sequence
        
        Args:
            problems: Dict of {problem_name: (hamiltonian, num_qubits)}
            verbose: Print progress
        
        Returns:
            Dict of {problem_name: OptimizationResult}
        
        Example:
            >>> optimizer = UnifiedQAdaPruneOptimizer()
            >>> problems = {
            ...     'KNAPSACK': (H_knapsack, 4),
            ...     'TSP': (H_tsp, 5),
            ...     'SET_COVER': (H_set_cover, 3)
            ... }
            >>> results = optimizer.optimize_multiple(problems)
            >>> for name, result in results.items():
            ...     print(f"{name}: {result.final_cost:.6f}")
        """
        
        results = {}
        total = len(problems)
        
        for i, (name, (hamiltonian, num_qubits)) in enumerate(problems.items(), 1):
            if verbose:
                print(f"\n[{i}/{total}] {name}...")
            
            result = self.optimize(
                problem_name=name,
                hamiltonian=hamiltonian,
                num_qubits=num_qubits,
                verbose=verbose
            )
            results[name] = result
        
        return results
    
    def get_summary(self) -> str:
        """Get summary of all optimizations"""
        if not self.results_history:
            return "No optimizations performed yet"
        
        summary = "\n" + "="*60
        summary += "\nOPTIMIZATION SUMMARY\n"
        summary += "="*60 + "\n"
        summary += f"{'Problem':<20} {'Cost':<15} {'Params':<12} {'Time':<10}\n"
        summary += "-"*60 + "\n"
        
        for result in self.results_history:
            summary += f"{result.problem_name:<20} "
            summary += f"{result.final_cost:<15.6f} "
            summary += f"{result.num_parameters:<12} "
            summary += f"{result.total_time:<10.3f}s\n"
        
        summary += "="*60 + "\n"
        
        return summary