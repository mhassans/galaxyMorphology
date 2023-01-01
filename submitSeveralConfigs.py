import os
import sys
from funcs import produceConfig, setConfigName

def main(submitToBatch):
    #initialize config
    config = dict(
        load_model = False,
        class_weight = None,
        classical = True,
        gamma = 'scale',
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
    
    #for clfType in [True, False]:
    #    config['classical'] = clfType
    #for Cclass in [0.1, 1.0, 10, 100]:
    for Cclass in [1000]:
    #for Cclass in [1.0e+4, 1.0e+5, 1.0e+6, 1.0e+7, 1.0e+8]:
        config['C_class'] = Cclass
        for i in range(config['n_splits']):
            config['fold_idx'] = i
            fileName = setConfigName(config)
            filePath = produceConfig(config, fileName)
            if (submitToBatch):
                os.system('qsub -v input="' + filePath + '" glxMorph.sh')
            else:
                os.system('python main.py '+ filePath + '> log/' + fileName + '.out')

if __name__ == "__main__":
    submitToBatch = False
    main(submitToBatch)
