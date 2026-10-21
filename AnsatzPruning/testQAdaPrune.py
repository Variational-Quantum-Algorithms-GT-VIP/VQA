# test_qadaprune.py
"""
Quick test to verify QAdaPrune is working
"""

print("Testing QAdaPrune installation...")

# Test 1: Can we import?
print("\n✓ Step 1: Testing imports...")
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import StatevectorEstimator as Estimator
    print("  ✓ Qiskit imports OK")
except ImportError as e:
    print(f"  ✗ Qiskit import failed: {e}")
    exit(1)

try:
    from QAdaPrune import QAdaPruner, QAdaPrunedCircuit
    print("  ✓ QAdaPrune imports OK")
except ImportError as e:
    print(f"  ✗ QAdaPrune import failed: {e}")
    exit(1)

try:
    from QAdaPrunedMomentumOptimizer import QAdaPrunedMomentumOptimizer, OptimizationConfig
    print("  ✓ QAdaPrunedMomentumOptimizer imports OK")
except ImportError as e:
    print(f"  ✗ QAdaPrunedMomentumOptimizer import failed: {e}")
    exit(1)

# Test 2: Can we create a simple optimization?
print("\n✓ Step 2: Testing basic optimization...")
try:
    # Simple Hamiltonian
    H = SparsePauliOp.from_list([
        ("ZIZZ", 7),
        ("ZZII", 1),
        ("IZZI", 5),
        ("IIZZ", 1)
    ])
    
    # Create circuit
    circuit = QuantumCircuit(4)
    estimator = Estimator()
    
    # Run optimization
    optimizer = QAdaPrunedMomentumOptimizer(
        circuit=circuit,
        hamiltonian=H,
        estimator=estimator,
        config=OptimizationConfig(max_layers=2)  # Just 2 layers for quick test
    )
    
    print("  Running optimization...")
    results = optimizer.optimize(verbose=False)
    
    print(f"  ✓ Optimization completed!")
    print(f"    - Final cost: {results['final_cost']:.6f}")
    print(f"    - Parameters: {results['num_parameters']}")
    print(f"    - Layers: {results['num_layers']}")
    print(f"    - Time: {results['total_time']:.3f}s")
    
except Exception as e:
    print(f"  ✗ Optimization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("✓ All tests passed! QAdaPrune is working!")
print("="*60)