import numpy as np
import math
import time
import pandas as pd
from random import randint
import matplotlib
# matplotlib.use('MacOSX')
# matplotlib.rcParams['interactive'] == True
from scipy.optimize import minimize
from qiskit_aer.aerprovider import AerSimulator
#from qiskit_aer import Aer, aerprovider
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.library import *
from qiskit.circuit import ClassicalRegister, QuantumRegister, Parameter
from qiskit.primitives import BackendEstimatorV2
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.transpiler import StagedPassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import *
from typing import overload

#H = SparsePauliOp.from_list([("ZIZZ", 1),("ZZII", 1),("IZZI", 2),("IIZZ", 1)])

# Generic cost function for hamiltonian
@overload
def cost_func(params, circuit, hamiltonian, estimator:Estimator): ...

@overload
def cost_func(params, circuit, hamiltonian, estimator:BackendEstimatorV2, pass_manager:StagedPassManager, isa_cache=None): ...

def cost_func(params, circuit, hamiltonian, estimator, pass_manager=None, isa_cache=None):
    #circuit = transpile(circuit)
    #circuit = circuit.assign_parameters(params)
    if pass_manager is not None:
        pub = to_isa_pub(circuit, hamiltonian, params, pass_manager, isa_cache)
    else:
        pub = (circuit, hamiltonian, params)
    cost = estimator.run([pub]).result()[0].data.evs
    """ for (obs, expval) in zip(hamiltonian, cost):
        if obs is H:
            print('a')
            return expval """
    return cost


# Parameter-shift rule gradiant calculator for ith parameter
@overload
def gradi(i, params, circuit, hamiltonian, estimator:Estimator): ...

@overload
def gradi(i, params, circuit, hamiltonian, estimator:BackendEstimatorV2, pass_manager:StagedPassManager, isa_cache=None): ...

def gradi(i, params, circuit, hamiltonian, estimator, pass_manager=None, isa_cache=None):
    delta = np.zeros(len(params))
    delta[i] = math.pi/2
    if pass_manager is not None:
        costp = cost_func(params+delta, circuit, hamiltonian, estimator, pass_manager, isa_cache)
        costm = cost_func(params-delta, circuit, hamiltonian, estimator, pass_manager, isa_cache)
    else:
        costp = cost_func(params+delta, circuit, hamiltonian, estimator)
        costm = cost_func(params-delta, circuit, hamiltonian, estimator)
    return (costp-costm)/2


# Hardware layout mapper and compiler with reference caching for V2 Primitive Pubs
def to_isa_pub(circuit:QuantumCircuit, observables, params:list, pass_manager:StagedPassManager, isa_cache:dict=None):
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