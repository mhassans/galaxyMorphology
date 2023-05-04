import subprocess
import sys
from funcs import produceConfig, setConfigName
from funcs import dataMap_custom1, dataMap_custom2, dataMap_custom3, dataMap_custom4

def run(config, submitToBatch):
    fileName = setConfigName(config)
    filePath = produceConfig(config, fileName)
    maxRunTime = 500 #in seconds. Only applies when running locally (i.e. not when submitted to the batch)
    if (submitToBatch):
        subprocess.run(['qsub', '-v', 'input='+filePath, 'glxMorph.sh'])
    else:
        try:
            with open('log/' + fileName + '.out', 'w') as f:
                subprocess.run(['python', 'main.py', filePath], stdout=f, timeout=maxRunTime)
        except subprocess.TimeoutExpired:
            print('TOOK LONG TO RUN THE FOLLOWING FILE. TERMINATED. NEEDS BEING SUBMITTED TO THE BATCH:', fileName)
            print('*************************************************************')
            with open("longJobs6.txt", "a") as file: #add the name of the terminated job to the end of this txt file.
                file.write(fileName)

def main(submitToBatch):
    #initialize config. Some values will be changed later in this script.
    config = dict(
        load_model = False,
        class_weight = None,
        classical = False,
        gamma = 'auto',
        C_class = 1.0e+6,
        alpha = 0.1,
        C_quant = 1.0e+6,
        data_map_func = None,
        interaction = ['Z', 'YY'],
        circuit_width = 5,
        trainPlusTestSize = 125,
        n_splits = 5,
        fold_idx = 0,
        minOfK = 5,
        modelSavedPath = 'trainedModels/',
        resultOutputPath = 'output/'
    )
    #list of configs to iterate over 
    list_minOfK = [5]#[5, 10, 20]
    list_trainPlusTestSize = [25000]
    list_classical = [False] #e.g. [True, False]
    list_weight = [None]#['balanced'] #e.g. [None, 'balanced']
    list_fold_idx = list(range(config['n_splits'])) # run over all folds
    
    #Classical-only lists to iterate over
    list_C_class = [1.0e+8]#[100, 1000, 1.0e+4, 1.0e+5, 1.0e+6, 1.0e+7, 1.0e+8]
    list_gamma = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100, 'auto', 'scale']
    
    #Introduce some auxiliary variables that is sometimes helpful for defining list_interaction below
    singleQubitInt = ['X', 'Y', 'Z']
    twoQubitInt = [first + second for first in singleQubitInt for second in singleQubitInt] # create this list: ['XX', 'XY', ...]
    singleThenTwoQubitInt = [[a,b] for a in singleQubitInt for b in twoQubitInt] # create this list: [['X', 'XX'], ['X','XY'], ...]
    
    #Quantum-only lists to iterate over
    list_alpha = [0.03]#[0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.13, 0.5]
    list_C_quant = [1.0e+7]#[1000, 1.0e+5]#[10, 1000, 1.0e+5, 1.0e+6]
    list_data_map_func = [None]#[dataMap_custom1, dataMap_custom2, dataMap_custom3, dataMap_custom4]
    list_interaction = singleThenTwoQubitInt #[['Z', 'ZX']]#a subset of singleThenTwoQubitInt
   
    for minOfK in list_minOfK:
        config['minOfK'] = minOfK
        for trainPlusTestSize in list_trainPlusTestSize:
            config['trainPlusTestSize'] = trainPlusTestSize
            for clfType in list_classical:
                config['classical'] = clfType
                for weight in list_weight:
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
                                    for data_map_func in list_data_map_func:
                                        config['data_map_func'] = data_map_func
                                        for interaction in list_interaction:
                                            config['interaction'] = interaction
                                            run(config, submitToBatch) #run

if __name__ == "__main__":
    submitToBatch = True
    main(submitToBatch)
