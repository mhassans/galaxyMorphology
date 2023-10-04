import numpy as np
import sys
import os
from pathlib import Path
from sklearn.svm import SVC
#import joblib
import time
from funcs import get_corr

from qiskit.circuit.library import PauliFeatureMap
from qiskit import QuantumCircuit

#packages for simulation
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

#packages for running on a quantum device
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit_ibm_runtime import Sampler, QiskitRuntimeService, Options
from qiskit_machine_learning.kernels import FidelityQuantumKernel

class QKE_SVC():
    """
    Defines an SVC classifier model family - either classical or quantum-enhanced.
    Can be used to train/load a given instance of the family.
    Classical case is an rbf.
    Quantum is Havlicek-based (IPQ).
    """
    def __init__(self,
                 classical,
                 class_weight, 
                 modelSavedPath,
                 entangleType,
                 nShots,
                 RunOnIBMdevice,
                 corrMethod,
                 gamma = None, 
                 C_class = None, 
                 alpha = None,
                 alphaCorr = None,
                 C_quant = None,
                 data_map_func = None,
                 interaction = None,
                 circuit_width = None):
        def getOptions(numShots, layout):
            """
            Used when running on a real device. Creates an Options() instance, adds properties, and returns it.
            """
            options = Options()
            options.optimization_level = 3 #Error suppression. 3 is the highest optimisation level (and the default)
            options.resilience_level = 1 #Error mitigation. 3 is the highest but Sampler supports up to 1.
            options.transpilation.initial_layout = layout #List of physical qubits for mapping virtual qubits to them. 
                                                                #e.g. [6,13] means 0->6 & 1->13, where the RHS show physical qubits.
            if numShots is not None:
                options.execution.shots = numShots #When None, the default (4000) is used.
            return options
        def corr_circ_pair(corr=0.0, alphaCorr=0.03):
            """
            circuit for implementing correlation between pair of feautures given their correlation and rotation factor alphaCorr. 
            The actual circuit is exp(alphaCorr*corr*ZZ), where ZZ applies on the two qubit in the pair.
            """
            circ = QuantumCircuit(2)
            circ.cx(0,1)
            circ.p(alphaCorr*corr, 1)
            circ.cx(0,1)
            #circ.barrier()    
            return circ
        def add_corr(main_circuit, alphaCorr, entanglement='full', method='pearson'):
            """
            Gets main_circuit and adds corr_circ_pair between each pair of qubits before main_circuit
            """
            indices = [0, 1, 2, 3, 4]
            if entanglement=='full':
                for i in reversed(range(len(indices))):
                    for j in reversed(range(i + 1, len(indices))):
                        main_circuit = main_circuit.compose(corr_circ_pair(corr=get_corr(i, j, method=method), alphaCorr=alphaCorr), qubits=[i,j], front=True)
            elif entanglement=='linear':
                for i in reversed(range(len(indices)-1)):
                    main_circuit = main_circuit.compose(corr_circ_pair(corr=get_corr(i, i+1, method=method), alphaCorr=alphaCorr), qubits=[i,i+1], front=True)
            else:
                raise ValueError("entanglement must be 'full' or 'linear'")
            h_layer = QuantumCircuit(len(indices))
            h_layer.h(range(len(indices)))
            main_circuit = main_circuit.compose(h_layer, front=True)
            return main_circuit
        self.classical = classical
        self.class_weight = class_weight
        self.modelSavedPath = modelSavedPath
        self.cache_chosen = 1000
        if classical:
            self.gamma = gamma
            self.C_class = C_class
        else:
            self.C_quant = C_quant
            pauliMap = PauliFeatureMap(circuit_width, alpha=alpha, paulis=interaction,\
                                            data_map_func=data_map_func, entanglement=entangleType)
            feature_map = add_corr(pauliMap, alphaCorr=alphaCorr, entanglement=entangleType, method=corrMethod)
            if(RunOnIBMdevice):
                service = QiskitRuntimeService(channel="ibm_quantum")
                backend = service.backend("ibmq_manila")#"ibm_nairobi")
                layout = [0,1,2,3,4]#[0,1,3,5,6] # List of physical qubits for mapping virtual qubits to them.
                options = getOptions(nShots, layout)
                sampler = Sampler(session=backend, options=options)
                fidelity = ComputeUncompute(sampler=sampler)
                self.kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
            else:
                self.kernel = FidelityStatevectorKernel(feature_map=feature_map, shots=nShots)
                if nShots is not None:
                    print('===============\n Note: Ignore ComplexWarning. When nShots is finite, forcing kernel mtx to'+ \
                           'be semidefinite (which is True by default in FidelityStatevectorKernel) could produce complex'+ \
                           'values in the matrix. From qiskit doc: "due to truncation and rounding errors we may get'+ \
                           'complex numbers". So the imaginary values should be small and hence ignored. \n===============')

    def train_model(self, train_data, train_labels, fileName):
        if self.classical:
            model = SVC(kernel = 'rbf', 
                        gamma = self.gamma,
                        C = self.C_class,
                        cache_size = self.cache_chosen,
                        class_weight = self.class_weight)
        else:
            model = SVC(kernel = self.kernel.evaluate,
                        C = self.C_quant,
                        cache_size = self.cache_chosen,
                        class_weight = self.class_weight)
        model.fit(train_data, train_labels)
        #save fitted SVC model
        filename = self.modelSavedPath + '/model_'+fileName+'.sav'
        if not Path(self.modelSavedPath).exists():
            Path(self.modelSavedPath).mkdir(parents=True)
        time0 = time.time()
        #joblib.dump(model, filename) #FIXME: Uncomment it. Currently gives error and does not save the model.
        print('SVC model trained and stored as:', filename)
        print("Storing model on disk took ", time.time()-time0," seconds")
        return model

    def set_model(self, load, model = None, train_data = None, train_labels = None, fileName = None):
        if load:
            self.model = model
            print('model has been loaded, model: ', self.model)
        else:
            self.model = self.train_model(train_data = train_data, train_labels = train_labels, fileName = fileName)

    def test(self, test_data):
        decFunc = self.model.decision_function(test_data)
        predicts = (decFunc > 0).astype(int) #instead of self.model.predict(test_data) -> more efficient coding for quantum device.
        return predicts, decFunc
