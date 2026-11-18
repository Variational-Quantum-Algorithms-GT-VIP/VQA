import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import *
from qiskit.primitives import BackendEstimatorV2 as BackendEstimator
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.transpiler import StagedPassManager as PassManager
from qiskit.quantum_info import *
from typing import overload

# Generic cost function for hamiltonian
@overload
def cost_func(params, circuit, hamiltonian, estimator:Estimator): ...

@overload
def cost_func(params, circuit, hamiltonian, estimator:BackendEstimator, pass_manager:PassManager, isa_cache=None): ...

def cost_func(params, circuit, hamiltonian, estimator, pass_manager=None, isa_cache=None):
    """
    Args:
        params: Parameter values (numpy array)
        circuit: Quantum circuit
        hamiltonian: SparsePauliOp Hamiltonian
        estimator: StatevectorEstimator | BackendEstimatorV2
        pass_manager: StagedPassManager (for BackendEstimatorV2 only)
        isa_cache: dict (for BackendEstimatorV2 only)
    
    Returns:
        float: Cost value (or inf if error)
    """

    try:
        # Handle empty parameters
        if len(params) == 0 or circuit.num_parameters == 0:
            params_to_use = np.array([])
        else:
            params_to_use = np.array(params).flatten()
        
        # Create publication
        if pass_manager is not None:
            pub = to_isa_pub(circuit, hamiltonian, params_to_use, pass_manager, isa_cache)
        else:
            pub = (circuit, hamiltonian, params_to_use)
        
        # Get result
        result = estimator.run([pub]).result()[0].data.evs
        
        # Handle different return types
        if isinstance(result, (list, tuple, np.ndarray)) or hasattr(result, "__len__"):
            flat_result = np.ravel(result)
            cost = float(flat_result[-1]) if len(flat_result) > 0 else float('inf')
        else:
            cost = float(result)
        
        # Validate result
        if np.isnan(cost) or np.isinf(cost):
            return float('inf')
        
        return cost
        
    except Exception as e:
        print(f"Warning: cost_func error: {e}")
        return float('inf')


# Parameter-shift rule gradiant calculator for ith parameter
@overload
def gradi(i, params, circuit, hamiltonian, estimator:Estimator, delta=np.pi/2): ...

@overload
def gradi(i, params, circuit, hamiltonian, estimator:BackendEstimator, pass_manager:PassManager, isa_cache=None, delta=np.pi/2): ...

def gradi(i, params, circuit, hamiltonian, estimator, pass_manager=None, isa_cache=None, delta=np.pi/2):
    try:
        params = np.array(params).flatten()
        
        # Forward shift
        params_plus = params.copy()
        params_plus[i] += delta
        
        # Backward shift
        params_minus = params.copy()
        params_minus[i] -= delta
        
        # Compute costs
        cost_plus = cost_func(params_plus, circuit, hamiltonian, estimator, pass_manager, isa_cache)
        cost_minus = cost_func(params_minus, circuit, hamiltonian, estimator, pass_manager, isa_cache)
        
        # Compute gradient
        if np.isinf(cost_plus) or np.isinf(cost_minus):
            return 0.0
        
        gradient = (cost_plus - cost_minus) / (2 * delta)
        
        return float(gradient)
        
    except Exception as e:
        print(f"Warning: gradi error: {e}")
        return 0.0


# Hardware layout mapper and compiler with reference caching for V2 Primitive Pubs
def to_isa_pub(circuit:QuantumCircuit, observables, params:list, pass_manager:PassManager, isa_cache:dict=None):
    if isa_cache is None:
        isa_cache = {}
    key = id(circuit)
    cached = isa_cache.get(key)
    if cached is None or cached[0] is not circuit:
        isa_circuit = pass_manager.run(circuit)
        cached = (circuit, isa_circuit)
        isa_cache[key] = cached
    _, isa_circuit = cached
    is_list = isinstance(observables, (list, tuple))
    obs_list = list(observables) if is_list else [observables]
    isa_observables_list = [
        (obs if isinstance(obs, SparsePauliOp) else SparsePauliOp(obs)).apply_layout(isa_circuit.layout)
        for obs in obs_list
    ]
    isa_observables = isa_observables_list if is_list else isa_observables_list
    return isa_circuit, isa_observables, params