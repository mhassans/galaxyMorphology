import os
import sys
from funcs import produceConfig, setConfigName

def run(config, submitToBatch):
    fileName = setConfigName(config)
    filePath = produceConfig(config, fileName)
    if (submitToBatch):
        os.system('qsub -v input="' + filePath + '" glxMorph.sh')
    else:
        os.system('python main.py '+ filePath + '> log/' + fileName + '.out')

def main(submitToBatch):
    #initialize config
    config = dict(
        load_model = False,
        class_weight = None,
        classical = True,
        gamma = 'auto',
        C_class = 1.0e+6,
        alpha = 0.1,
        C_quant = 1.0e+6,
        single_mapping = 1,
        pair_mapping = 1,
        interaction = 'YY',
        circuit_width = 7,
        trainPlusTestSize = 25000,
        n_splits = 5,
        fold_idx = 0,
        modelSavedPath = 'trainedModels/',
        resultOutputPath = 'output/'
    )
   
    list_classical = [True, False]
    list_class_weight = [None, 'balanced']
    list_fold_idx = list(range(config['n_splits']))
    
    #Classic-only
    list_C_class = [0.1, 1.0, 10, 100, 1000, 1.0e+4, 1.0e+5, 1.0e+6, 1.0e+7, 1.0e+8]
    list_gamma = [1, 'scale', 'auto']
    
    #Quantum-only
    list_alpha = [0.1]
    list_C_quant = [1.0e+6]
    list_single_mapping = [1]
    list_pair_mapping = [1]
    list_interaction = ['YY']
    
    for clfType in list_classical:
        config['classical'] = clfType
        for weight in list_class_weight:
            config['class_weight'] = weight
            for foldID in list_fold_idx:
                config['fold_idx'] = foldID
                
                if config['classical']:
                    for Cclass in list_C_class:
                        config['C_class'] = Cclass
                        for gamma in list_gamma:
                            config['gamma'] = gamma
                            run(config, submitToBatch) #run
                else:
                    for alpha in list_alpha:
                        config['alpha'] = alpha
                        for CQuant in list_C_quant:
                            config['C_quant'] = CQuant
                            for snglMap in list_single_mapping:
                                config['single_mapping'] = snglMap
                                for pairMap in list_pair_mapping:
                                    config['pair_mapping'] = pairMap
                                    for interaction in list_interaction:
                                        config['interaction'] = interaction
                                        run(config, submitToBatch) #run

if __name__ == "__main__":
    submitToBatch = False
    main(submitToBatch)
