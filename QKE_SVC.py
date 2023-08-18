import numpy as np
import sys
import os
from pathlib import Path
from sklearn.svm import SVC
#import joblib
import time
import pickle

from qiskit.circuit.library import PauliFeatureMap

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
                 savedModelPath,
                 savedKernelPath,
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
        self.classical = classical
        self.class_weight = class_weight
        self.savedModelPath = savedModelPath
        self.savedKernelPath = savedKernelPath
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
                backend = service.backend("ibm_nairobi")
                layout = [0,1,3,5,6] # List of physical qubits for mapping virtual qubits to them.
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

    
    def calc_train_kernel(self, train_data, fileName):
        print('Computing kernel matrix for train data:')
        self.kernelMtxTrain = self.kernel.evaluate(x_vec=train_data)
    def calc_train_kernel(self, train_data, fileName):
        kernelDict = {}
        kernelDict['train_labels'] = train_labels
        kernelDict['test_labels'] = test_labels
        print('Computing kernel matrix for train data:')
        kernelMtxTrain = self.kernel.evaluate(x_vec=train_data)
        print('Computing kernel matrix for test data:')
        kernelMtxTest = self.kernel.evaluate(x_vec=train_data, y_vec=test_data)
        kernelDict['kernelMtxTrain'] = kernelMtxTrain
        kernelDict['kernelMtxTest'] = kernelMtxTest
        print('Saving Kernels')
        #Adding time.time() to the name to produce a unique name. Helpful for later when circuits with low number of shots are merged.
        fullFileName = self.savedKernelPath + '/kernel_'+fileName+str(time.time()).replace('.','')+'.pkl' 
        if not Path(self.savedKernelPath).exists():
            Path(self.savedKernelPath).mkdir(parents=True)
        with open(fullFileName, 'wb') as f:
            pickle.dump(kernelDict, f)
        
    def train_model(self, train_data, train_labels, fileName):
        if self.classical:
            model = SVC(kernel = 'rbf', 
                        gamma = self.gamma,
                        C = self.C_class,
                        cache_size = self.cache_chosen,
                        class_weight = self.class_weight)
            model.fit(train_data, train_labels)
        else:
            model = SVC(kernel = 'precomputed',
                        C = self.C_quant,
                        cache_size = self.cache_chosen,
                        class_weight = self.class_weight)
            calc_train_kernel()
            model.fit(getTrainMtx(fileName), train_labels)
        #save fitted SVC model
        filename = self.savedModelPath + '/model_'+fileName+'.sav'
        if not Path(self.savedModelPath).exists():
            Path(self.savedModelPath).mkdir(parents=True)
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
