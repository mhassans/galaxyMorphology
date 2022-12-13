"""
Use data provided to train and save an svm classifier
Can use a classicla svm or quantum-enhanced
STATUS: in dev, job report could be compiled in main
"""
print('142')
import numpy as np
print('142')
import sys
print('143')
import os
print('144')
from pathlib import Path
print('145')
from sklearn.svm import SVC
print('146')
import joblib
print('147')

from qiskit import (Aer,IBMQ)
print('148')
IBMQ.load_account()
print('149')
IBMQ.providers()
print('1410')
provider = IBMQ.get_provider(group='open')
print('1411')
from qiskit.utils import QuantumInstance
print('1412')
from qiskit.circuit import ParameterVector
print('1413')
from qiskit_machine_learning.kernels import QuantumKernel
print('1414')

path_for_imports = os.path.abspath('.')
print(path_for_imports)
sys.path.append(path_for_imports) #for quantum_circuit
from quantum_circuit import (param_feature_map, param_U_gate)

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
    gamma = None, 
    C_class = None, 
    alpha = None,
    C_quant = None,
    single_mapping = None,
    pair_mapping = None,
    interaction = None,
    circuit_width = None):
        if classical:
            self.gamma = gamma
            self.C_class = C_class
        else:
            self.alpha = alpha
            self.C_quant = C_quant
            self.single_mapping = single_mapping
            self.pair_mapping = pair_mapping
            self.interaction = interaction

            self.backend = QuantumInstance(Aer.get_backend('statevector_simulator'))
            self.circuit_width = circuit_width
            params = ParameterVector('phi', circuit_width)
            U_gate = param_U_gate.U_flexible(circuit_width, 
            params, 
            single_mapping = self.single_mapping, 
            pair_mapping = self.pair_mapping, 
            interaction = self.interaction, 
            alpha = self.alpha) 
            featureMap = param_feature_map.feature_map(circuit_width, U_gate)

            self.kernel = QuantumKernel(feature_map = featureMap, quantum_instance = self.backend)
        self.classical = classical
        self.class_weight = class_weight
        self.modelSavedPath = modelSavedPath
        self.cache_chosen = 1000

    def train_model(self, train_data, train_labels, fileName):
        if self.classical:
            model = SVC(kernel = 'rbf', 
            gamma = self.gamma,
            C = self.C_class,
            cache_size = self.cache_chosen,
            class_weight = self.class_weight) #TODO: Try 'balanced'?
            model.fit(train_data, train_labels)
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
        joblib.dump(model, filename)
        print('SVC model trained and stored as:', filename)
        return model

    def set_model(self, load, model = None, train_data = None, train_labels = None, fileName = None):
        if load:
            self.model = model
            print('model has been loaded, model: ', self.model)
        else:
            self.model = self.train_model(train_data = train_data, train_labels = train_labels, fileName = fileName)

    def test(self, test_data):
        return self.model.predict(test_data), self.model.decision_function(test_data)
