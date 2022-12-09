import os
import sys
from funcs import produceConfig, getConfigName


def main():
    #os.system('rm config/autoConfigs_*') #remove previous configs to ensure new configs are used
    
    #initialize config
    config = dict(
        load_model = False,
        class_weight = 'balanced',
        classical = False,
        gamma = 1,
        C_class = 1.0e+6,
        alpha = 0.1,
        C_quant = 1.0e+6,
        single_mapping = 1,
        pair_mapping = 1,
        interaction = 'YY',
        circuit_width = 7,
        trainPlusTestSize = 200,
        testSetSize = 0.5,
        modelSavedPath = 'trainedModels/',
        resultOutputPath = 'output/'
    )
    
    for clfType in [True, False]:
        config['classical'] = clfType
        fileName = getConfigName(config)
        filePath = produceConfig(config, fileName)
        os.system('qsub -v input="' + filePath + '" glxMorph.sh')
        #os.system('python main.py '+ filePath)

if __name__ == "__main__":
    main()
