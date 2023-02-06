import subprocess
import sys
from funcs import produceConfig, setConfigName

def run(config, submitToBatch):
    fileName = setConfigName(config)
    filePath = produceConfig(config, fileName)
    maxRunTime = 2000 #seconds
    if (submitToBatch):
        subprocess.run(['qsub', '-v', 'input='+filePath, 'glxMorph.sh'])
    else:
        try:
            with open('log/' + fileName + '.out', 'w') as f:
                subprocess.run(['python', 'main.py', filePath], stdout=f, timeout=maxRunTime)
        except subprocess.TimeoutExpired:
            print('TOOK LONG TO RUN THE FOLLOWING FILE. TERMINATED. NEEDS BEING SUBMITTED TO THE BATCH:', fileName)
            print('*************************************************************')
            with open("longJobs.txt", "a") as file: #add the name of the terminated job to the end of this txt file.
                file.write(fileName)

def main(submitToBatch):
    #initialize config
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
        circuit_width = 7,
        trainPlusTestSize = 25000,
        n_splits = 5,
        fold_idx = 0,
        modelSavedPath = 'trainedModels/',
        resultOutputPath = 'output/'
    )
    #list of configs to iterate over 
    list_classical = [False] #e.g. [True, False]
    list_weight = [None] #e.g. [None, 'balanced']
    list_fold_idx = list(range(config['n_splits'])) # run over all folds
    
    #Classical-only
    list_C_class = [10] #e.g. [0.1, 1.0, 10, 100, 1000, 10000]
    list_gamma = [1] #e.g. [0.1, 1, 'scale', 'auto']
    
    #Quantum-only
    singleQubitInt = ['X', 'Y', 'Z']
    twoQubitInt = [first + second for first in singleQubitInt for second in singleQubitInt] # create this list: ['XX', 'XY', ...]
    singleThenTwoQubitInt = [[a,b] for a in singleQubitInt for b in twoQubitInt] # create this list: [['X', 'XX'], ['X','XY'], ...]
    list_alpha = [0.2, 1.2, 2]#[0.2, 0.6, 1.2, 1.6, 2]
    list_C_quant = [10, 1000, 1.0e+6]#[1.0, 10, 100, 1000, 1.0e+4, 1.0e+5, 1.0e+6]
    list_data_map_func = [None]    
    list_interaction = [item for item in singleThenTwoQubitInt if item[1] == 'YY'] #could use other singleThenTwoQubitInt elements
    
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
