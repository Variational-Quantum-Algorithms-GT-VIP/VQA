# visualization.py
"""
Visualization utilities for QAdaPrune results
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from UnifiedQAdaPruneOptimizer import OptimizationResult


def plot_single_optimization(result: OptimizationResult, save_path: str = None):
    """
    Plot a single optimization result
    
    Args:
        result: OptimizationResult object
        save_path: Where to save (optional)
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{result.problem_name} - QAdaPrune Optimization', fontsize=16)
    
    history = result.history
    
    # Plot 1: Cost evolution
    ax = axes[0, 0]
    ax.plot(history['costs'], 'o-', linewidth=2, markersize=8, color='blue')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cost (Energy)')
    ax.set_title('Cost Evolution')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Parameter count evolution
    ax = axes[0, 1]
    ax.plot(history['param_counts'], 's-', linewidth=2, markersize=8, color='green', label='Active')
    ax.bar(range(len(history['layer_sizes'])), history['layer_sizes'], 
           alpha=0.5, color='orange', label='New per layer')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Parameter Count')
    ax.set_title('Parameter Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Pruning progress
    ax = axes[1, 0]
    ax.bar(range(len(history['pruned_counts'])), history['pruned_counts'], color='red', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cumulative Pruned Parameters')
    ax.set_title('Pruning Progress')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Gradient convergence
    ax = axes[1, 1]
    ax.semilogy(history['gradients'], 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Max Gradient Magnitude (log scale)')
    ax.set_title('Convergence')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to: {save_path}")
    
    plt.show()


def plot_multi_problem_comparison(results: Dict[str, OptimizationResult], 
                                  save_path: str = None):
    """
    Compare multiple problem results
    
    Args:
        results: Dict of {problem_name: OptimizationResult}
        save_path: Where to save (optional)
    """
    
    problems = list(results.keys())
    costs = [results[p].final_cost for p in problems]
    params = [results[p].num_parameters for p in problems]
    times = [results[p].total_time for p in problems]
    pruned = [results[p].pruned_parameters for p in problems]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-Problem Comparison - QAdaPrune', fontsize=16)
    
    # Plot 1: Final Costs
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(problems)))
    ax.bar(problems, costs, color=colors, alpha=0.7)
    ax.set_ylabel('Final Cost')
    ax.set_title('Final Cost Comparison')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Parameter Counts
    ax = axes[0, 1]
    x = np.arange(len(problems))
    width = 0.35
    ax.bar(x - width/2, params, width, label='Active', color='green', alpha=0.7)
    ax.bar(x + width/2, pruned, width, label='Pruned', color='red', alpha=0.7)
    ax.set_ylabel('Parameter Count')
    ax.set_title('Parameter Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(problems, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Optimization Time
    ax = axes[1, 0]
    ax.bar(problems, times, color=colors, alpha=0.7)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Optimization Time')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary Table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [['Problem', 'Cost', 'Params', 'Time (s)']]
    for p in problems:
        table_data.append([
            p,
            f"{costs[problems.index(p)]:.4f}",
            f"{params[problems.index(p)]}",
            f"{times[problems.index(p)]:.2f}"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison plot to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage:
    # from UnifiedQAdaPruneOptimizer import UnifiedQAdaPruneOptimizer
    # optimizer = UnifiedQAdaPruneOptimizer()
    # result = optimizer.optimize(...)
    # plot_single_optimization(result, save_path="result.png")
    
    print("Visualization utilities loaded. Use plot_single_optimization() or plot_multi_problem_comparison()")