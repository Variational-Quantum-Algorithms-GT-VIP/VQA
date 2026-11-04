# run_all_tests.py
"""
Test QAdaPrune on all different problem types
Run this to test everything
"""

import csv
import time
from UnifiedQAdaPruneOptimizer import UnifiedQAdaPruneOptimizer, OptimizationConfig
from ProblemHandlers import (
    get_knapsack_problem,
    get_tsp_problem,
    get_maxcut_problem,
    get_setcover_problem,
    get_coloring_problem,
    KnapsackProblem,
    TSPProblem,
    MaxCutProblem,
    SetCoverProblem,
    GraphColoringProblem
)


def main():
    """Run tests on all problem types"""
    
    print("\n" + "="*70)
    print("QAdaPrune Multi-Problem Testing Framework")
    print("="*70)
    
    # Configuration
    config = OptimizationConfig(
        max_layers=3,
        pruning_threshold=0.01,
        patience=3,
        optimization_method='cobyla',
        max_iterations_per_layer=100
    )
    
    # Initialize optimizer
    optimizer = UnifiedQAdaPruneOptimizer(config=config)
    
    # Define all problems to test
    problems = {
        'KNAPSACK': get_knapsack_problem(),
        'TSP': get_tsp_problem(),
        'MAX_CUT': get_maxcut_problem(),
        'SET_COVER': get_setcover_problem(),
        'GRAPH_COLORING': get_coloring_problem(),
    }
    
    print(f"\nTesting {len(problems)} problems...")
    
    # Run all optimizations
    results = optimizer.optimize_multiple(problems, verbose=True)
    
    # Display summary
    print(optimizer.get_summary())
    
    # Save results to CSV
    print("\nSaving results to CSV...")
    csv_results = []
    for name, result in results.items():
        csv_results.append({
            'Problem': name,
            'Num_Qubits': result.num_qubits,
            'Final_Cost': result.final_cost,
            'Num_Layers': result.num_layers,
            'Num_Parameters': result.num_parameters,
            'Pruned_Parameters': result.pruned_parameters,
            'Total_Time': result.total_time,
            'Avg_Cost_per_Layer': np.mean(result.history['costs']) if result.history['costs'] else 0
        })
    
    # Write to CSV
    with open('qadaprune_multi_problem_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Problem', 'Num_Qubits', 'Final_Cost', 
                                                 'Num_Layers', 'Num_Parameters', 
                                                 'Pruned_Parameters', 'Total_Time',
                                                 'Avg_Cost_per_Layer'])
        writer.writeheader()
        writer.writerows(csv_results)
    
    print("✓ Results saved to: qadaprune_multi_problem_results.csv")
    
    # Performance summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    import numpy as np
    times = [r.total_time for r in results.values()]
    costs = [r.final_cost for r in results.values()]
    params = [r.num_parameters for r in results.values()]
    
    print(f"Average Time:       {np.mean(times):.3f}s ± {np.std(times):.3f}s")
    print(f"Average Cost:       {np.mean(costs):.6f}")
    print(f"Average Parameters: {np.mean(params):.1f}")
    print("="*70)


if __name__ == "__main__":
    import numpy as np
    main()