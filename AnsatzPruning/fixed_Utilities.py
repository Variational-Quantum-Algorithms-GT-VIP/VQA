# fixed_Utilities.py
"""
Fixed version of Utilities.py - use this instead of original
"""

import numpy as np
import math
import time
import pandas as pd
from random import randint
import csv
from scipy.optimize import minimize

from qiskit_aer.aerprovider import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.library import *
from qiskit.circuit import ClassicalRegister, QuantumRegister, Parameter
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import *


def cost_func(params, circuit, hamiltonian, estimator):
    """
    Fixed cost function - handles edge cases
    
    Args:
        params: Parameter values (numpy array)
        circuit: Quantum circuit
        hamiltonian: SparsePauliOp Hamiltonian
        estimator: StatevectorEstimator
    
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
        pub = (circuit, hamiltonian, params_to_use)
        
        # Get result
        result = estimator.run([pub]).result()[0].data.evs
        
        # Handle different return types
        if isinstance(result, (list, np.ndarray)):
            result = np.array(result)
            # Take last element (usually energy)
            if len(result) > 0:
                cost = float(result[-1])
            else:
                cost = float('inf')
        else:
            cost = float(result)
        
        # Validate result
        if np.isnan(cost) or np.isinf(cost):
            return float('inf')
        
        return cost
        
    except Exception as e:
        print(f"Warning: cost_func error: {e}")
        return float('inf')


def gradi(i, params, circuit, hamiltonian, estimator, delta=np.pi/2):
    """
    Parameter-shift rule gradient for ith parameter
    
    Args:
        i: Parameter index
        params: Parameter vector
        circuit: Quantum circuit
        hamiltonian: SparsePauliOp
        estimator: StatevectorEstimator
        delta: Shift amount
    
    Returns:
        float: Gradient value
    """
    try:
        params = np.array(params).flatten()
        
        # Forward shift
        params_plus = params.copy()
        params_plus[i] += delta
        
        # Backward shift
        params_minus = params.copy()
        params_minus[i] -= delta
        
        # Compute costs
        cost_plus = cost_func(params_plus, circuit, hamiltonian, estimator)
        cost_minus = cost_func(params_minus, circuit, hamiltonian, estimator)
        
        # Compute gradient
        if np.isinf(cost_plus) or np.isinf(cost_minus):
            return 0.0
        
        gradient = (cost_plus - cost_minus) / (2 * delta)
        
        return float(gradient)
        
    except Exception as e:
        print(f"Warning: gradi error: {e}")
        return 0.0