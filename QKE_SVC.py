import numpy as np
import sys
import os
from pathlib import Path
from sklearn.svm import SVC
#import joblib
import time

from qiskit.circuit.library import PauliFeatureMap

#packages for simulation
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

#packages for running on a quantum device
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit_ibm_runtime import Sampler, QiskitRuntimeService, Options
from qiskit_machine_learning.kernels import FidelityQuantumKernel

#from qiskit import (Aer,IBMQ)
#IBMQ.load_account()
#IBMQ.providers()
#provider = IBMQ.get_provider(group='open')

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
                 gamma = None, 
                 C_class = None, 
                 alpha = None,
                 C_quant = None,
                 data_map_func = None,
                 interaction = None,
                 circuit_width = None):
        self.classical = classical
        self.class_weight = class_weight
        self.modelSavedPath = modelSavedPath
        self.cache_chosen = 1000
        if classical:
            self.gamma = gamma
            self.C_class = C_class
        else:
            self.C_quant = C_quant
            feature_map = PauliFeatureMap(circuit_width, alpha=alpha, paulis=interaction,\
                                            data_map_func=data_map_func, entanglement=entangleType)        
            if(RunOnIBMdevice):
                service = QiskitRuntimeService(channel="ibm_quantum")
                backend = service.backend("ibmq_manila")
                options = Options(optimization_level=3, resilience_level=1)
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
        #joblib.dump(model, filename)
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
        return self.model.predict(test_data), self.model.decision_function(test_data)
