"""
QAdaPrune Integration Guide
Shows how to integrate QAdaPrune into your existing VQE optimization pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator

from QAdaPrunedMomentumOptimizer import (
    QAdaPrunedMomentumOptimizer,
    OptimizationConfig,
    MomentumTracker
)


# ============================================================================
# STEP 1: Basic Usage - Drop-in Replacement for Your Current Pipeline
# ============================================================================

def basic_usage_example():
    """
    Simplest way to use QAdaPrune - just replace your current optimizer
    """
    print("\n" + "="*70)
    print("STEP 1: Basic Usage")
    print("="*70)
    
    # Define your problem (same as before)
    H = SparsePauliOp.from_list([
        ("ZIZZ", 7),
        ("ZZII", 1),
        ("IZZI", 5),
        ("IIZZ", 1)
    ])
    
    # Setup circuit
    base_circuit = QuantumCircuit(4)
    estimator = Estimator()
    
    # Create optimizer with default config
    optimizer = QAdaPrunedMomentumOptimizer(
        circuit=base_circuit,
        hamiltonian=H,
        estimator=estimator
    )
    
    # Run optimization (replaces your LayerOptimizer + manual pruning)
    results = optimizer.optimize(verbose=True)
    
    return results


# ============================================================================
# STEP 2: Custom Configuration
# ============================================================================

def custom_config_example():
    """
    Advanced: Configure QAdaPrune for your specific problem
    """
    print("\n" + "="*70)
    print("STEP 2: Custom Configuration")
    print("="*70)
    
    H = SparsePauliOp.from_list([
        ("ZZIZ", 1),
        ("ZIZI", 1),
        ("IZZI", 1),
        ("ZZZI", 1),
        ("ZIZZ", 1)
    ])
    
    base_circuit = QuantumCircuit(4)
    estimator = Estimator()
    
    # Custom config for aggressive pruning
    config = OptimizationConfig(
        max_layers=6,
        pruning_threshold=0.02,  # More aggressive pruning
        momentum_beta1=0.95,      # Stronger momentum
        momentum_beta2=0.99,
        pruning_window_size=4,    # Shorter window = more responsive
        patience=4,               # Stop after 4 iterations of no improvement
        optimization_method="cobyla",  # or "de" for differential evolution
        max_iterations_per_layer=150
    )
    
    optimizer = QAdaPrunedMomentumOptimizer(
        circuit=base_circuit,
        hamiltonian=H,
        estimator=estimator,
        config=config
    )
    
    results = optimizer.optimize(verbose=True)
    
    return results


# ============================================================================
# STEP 3: Integration with Your Existing Code
# ============================================================================

def integration_with_existing_code():
    """
    How to integrate QAdaPrune with your MomentumBuilder, 
    OptimizerTest, and other existing code
    """
    print("\n" + "="*70)
    print("STEP 3: Integration with Existing Code")
    print("="*70)
    
    # BEFORE: You had to do this manually
    # from OptimizerTest import LayerOptimizer
    # from MomentumBuilder import MomentumBuilder
    # 
    # circuit = QuantumCircuit(4)
    # for layer in range(5):
    #     circuit, params = MomentumBuilder([1]*4, [0,1,2,3], circuit, ...)
    #     # Manual pruning with numpy/heapq...
    #     # Tracking parameters manually...
    
    # AFTER: Now just use QAdaPrunedMomentumOptimizer
    
    H = SparsePauliOp.from_list([
        ("ZZIII", 1),
        ("ZZIIZ", 1),
        ("IZZZZ", 1)
    ])
    
    # Your base circuit setup (same as before)
    n_qubits = 4
    circuit = QuantumCircuit(n_qubits)
    
    # Add initial state preparation if needed
    # circuit.h(range(n_qubits))  # For example
    
    estimator = Estimator()
    
    # One line replaces your entire pipeline!
    optimizer = QAdaPrunedMomentumOptimizer(
        circuit=circuit,
        hamiltonian=H,
        estimator=estimator,
        config=OptimizationConfig(max_layers=4)
    )
    
    results = optimizer.optimize(verbose=True)
    
    # Access results just like before
    optimized_circuit = results['optimized_circuit']
    params = results['optimized_params']
    cost = results['final_cost']
    
    print(f"\nIntegration complete!")
    print(f"Circuit: {optimized_circuit.num_qubits} qubits")
    print(f"Parameters: {len(params)}")
    print(f"Final cost: {cost:.6f}")
    
    return results


# ============================================================================
# STEP 4: Performance Analysis & Visualization
# ============================================================================

def analyze_optimization_history(results: Dict):
    """
    Analyze and visualize the optimization history
    """
    print("\n" + "="*70)
    print("STEP 4: Analyzing Optimization")
    print("="*70)
    
    history = results['history']
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cost evolution
    ax = axes[0, 0]
    ax.plot(history['costs'], 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cost (Energy)')
    ax.set_title('Cost Evolution')
    ax.grid(True, alpha=0.3)
    
    # Parameter count evolution
    ax = axes[0, 1]
    ax.plot(history['param_counts'], 'g-s', linewidth=2, markersize=6, label='Active')
    ax.plot(history['layer_sizes'], 'r--^', linewidth=2, markersize=6, label='New per layer')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Parameter Count')
    ax.set_title('Parameter Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Pruning progress
    ax = axes[1, 0]
    ax.bar(range(len(history['pruned_counts'])), history['pruned_counts'], color='orange')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cumulative Pruned Parameters')
    ax.set_title('Pruning Progress')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Gradient magnitude evolution
    ax = axes[1, 1]
    ax.semilogy(history['gradients'], 'purple', marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Max Gradient Magnitude (log scale)')
    ax.set_title('Gradient Convergence')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qadaprune_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved analysis plot to 'qadaprune_analysis.png'")
    
    return fig


# ============================================================================
# STEP 5: Comparison with Other Approaches
# ============================================================================

def comparison_example():
    """
    Compare QAdaPrune vs your existing methods
    """
    print("\n" + "="*70)
    print("STEP 5: Comparative Analysis")
    print("="*70)
    
    H = SparsePauliOp.from_list([
        ("ZZZZ", 1),
        ("ZIZZ", 0.5),
        ("IZZI", 0.3),
        ("IIZZ", 0.2)
    ])
    
    results_comparison = {
        'QAdaPrune': None,
        'Standard': None,
    }
    
    # QAdaPrune approach
    print("\n[1/2] Running QAdaPrune...")
    circuit = QuantumCircuit(4)
    optimizer_qada = QAdaPrunedMomentumOptimizer(
        circuit=circuit,
        hamiltonian=H,
        estimator=Estimator(),
        config=OptimizationConfig(max_layers=3)
    )
    results_comparison['QAdaPrune'] = optimizer_qada.optimize(verbose=False)
    
    # Standard approach (Momentum only, no pruning)
    print("[2/2] Running Standard Momentum (no pruning)...")
    circuit2 = QuantumCircuit(4)
    config_standard = OptimizationConfig(
        max_layers=3,
        pruning_threshold=float('inf')  # Disable pruning
    )
    optimizer_std = QAdaPrunedMomentumOptimizer(
        circuit=circuit2,
        hamiltonian=H,
        estimator=Estimator(),
        config=config_standard
    )
    results_comparison['Standard'] = optimizer_std.optimize(verbose=False)
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"{'Metric':<25} {'QAdaPrune':<20} {'Standard':<20}")
    print("-" * 65)
    
    for key in ['final_cost', 'num_parameters', 'pruned_parameters', 'total_time', 'num_layers']:
        qa_val = results_comparison['QAdaPrune'][key]
        std_val = results_comparison['Standard'][key]
        
        if isinstance(qa_val, float):
            qa_str = f"{qa_val:.6f}"
            std_str = f"{std_val:.6f}"
        else:
            qa_str = str(qa_val)
            std_str = str(std_val)
        
        print(f"{key:<25} {qa_str:<20} {std_str:<20}")
    
    print("="*70)
    
    return results_comparison


# ============================================================================
# STEP 6: Migration Guide - Adapting Your Existing Functions
# ============================================================================

def migration_guide_example():
    """
    Shows how to adapt your existing OptimizerTest.py and other modules
    """
    
    guide = """
    
    ============================================================================
    MIGRATION GUIDE: Converting Your Existing Code to QAdaPrune
    ============================================================================
    
    BEFORE (OptimizerTest.py):
    --------------------------
    def LayerOptimizer(params, ansatz, layers, circuit, hamiltonian, estimator):
        # Manual layer addition
        for l in range(layers):
            naiveLayer = QuantumCircuit(n)
            # Add RY gates...
            # Manual pruning with heapq...
            # Manual parameter tracking...
            x = minimize(cost_func, ...)
        return x, circuit, badcircuit
    
    
    AFTER (Using QAdaPrunedMomentumOptimizer):
    -------------------------------------------
    optimizer = QAdaPrunedMomentumOptimizer(
        circuit=circuit,
        hamiltonian=hamiltonian,
        estimator=estimator,
        config=OptimizationConfig(max_layers=layers)
    )
    results = optimizer.optimize()
    
    # Access results
    optimized_params = results['optimized_params']
    optimized_circuit = results['optimized_circuit']
    final_cost = results['final_cost']
    
    
    KEY DIFFERENCES:
    ----------------
    1. Automatic Layer Addition:
       - QAdaPrune uses momentum to select which qubits get new gates
       - No need to manually create QuantumCircuit objects
    
    2. Automatic Pruning:
       - Parameters tracked automatically via importance scores
       - No manual heapq operations needed
       - Pruning happens automatically when parameters are low-importance
    
    3. Better Convergence:
       - Converges automatically when no more improvement
       - Prevents overfitting by pruning irrelevant parameters
    
    4. Simpler API:
       - One config object instead of multiple parameters
       - One function call instead of nested loops
    
    5. Better Statistics:
       - Automatic tracking of costs, parameter counts, pruning history
       - Ready-to-plot history for visualization
    
    
    PARAMETER MAPPING:
    ------------------
    Old LayerOptimizer parameters → New config parameters:
    
    layers                          → config.max_layers
    (pruning rate hardcoded as 0.5) → config.pruning_threshold
    (no momentum before)             → config.momentum_beta1, beta2
    COBYLA method                   → config.optimization_method='cobyla'
    
    
    USING YOUR EXISTING UTILITIES:
    --------------------------------
    Your Utilities.py functions still work!
    - cost_func() is automatically called by QAdaPrunedMomentumOptimizer
    - gradi() is used internally for gradient calculations
    - No need to modify those files
    
    Just import what you need:
    from Utilities import cost_func, gradi
    
    """
    
    print(guide)


# ============================================================================
# STEP 7: Advanced - Custom Pruning Strategies
# ============================================================================

def advanced_custom_pruning():
    """
    Example of implementing custom pruning strategies
    """
    
    from QAdaPrune import QAdaPruner
    
    print("\n" + "="*70)
    print("STEP 7: Custom Pruning Strategies")
    print("="*70)
    
    # You can subclass QAdaPruner for custom behavior
    class AggressivePruner(QAdaPruner):
        """Prunes more aggressively based on multiple criteria"""
        
        def get_pruning_candidates(self):
            """Override to implement custom pruning logic"""
            candidates = []
            
            for param_id, metrics in self.parameters.items():
                if metrics.is_pruned:
                    continue
                
                # Custom criteria: prune if:
                # 1. Low importance score
                # 2. Low gradient variance (not sensitive to changes)
                # 3. In early layers (prefer to prune shallow qubits)
                
                max_importance = max(
                    m.importance_score for m in self.parameters.values() 
                    if not m.is_pruned
                )
                
                low_importance = metrics.importance_score < 0.02 * max_importance
                low_variance = metrics.gradient_variance < 1e-5
                early_layer = metrics.layer < 2
                
                if (low_importance and (low_variance or early_layer)):
                    candidates.append(param_id)
            
            return candidates
    
    print("✓ Custom pruning strategies can be implemented by subclassing QAdaPruner")
    print("  See AggressivePruner example above")


# ============================================================================
# MAIN: Run All Examples
# ============================================================================

if __name__ == "__main__":
    
    # Uncomment the examples you want to run:
    
    # Step 1: Basic usage
    results_1 = basic_usage_example()
    
    # Step 2: Custom config
    # results_2 = custom_config_example()
    
    # Step 3: Integration
    # results_3 = integration_with_existing_code()
    
    # Step 4: Analysis
    analyze_optimization_history(results_1)
    
    # Step 5: Comparison
    # results_comp = comparison_example()
    
    # Step 6: Migration guide
    migration_guide_example()
    
    # Step 7: Advanced
    # advanced_custom_pruning()
    
    print("\n" + "="*70)
    print("Integration Guide Complete!")
    print("="*70)