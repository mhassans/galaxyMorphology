import time
start_time = time.time()
import joblib
import numpy as np
import pandas as pd
from QKE_SVC import QKE_SVC
from funcs import load_config, parse_args, prepare_dataframe, get_train_test, normalize_data, makeOutputFileName
from pathlib import Path

config = load_config(parse_args().config)
fileName = makeOutputFileName(config['classical'],
                              config['trainPlusTestSize']*(1-config['testSetSize'])
                             )

df = prepare_dataframe(trainPlusTestSize=config['trainPlusTestSize'])
train_data, train_labels, test_data, test_labels = get_train_test(df, testSetSize=config['testSetSize'])
train_data = normalize_data(train_data)
test_data = normalize_data(test_data)

n_features = np.shape(train_data)[1]
if not (config['circuit_width'] == n_features):
    raise AssertionError('Number of feautures is: ', n_features, '.',
    ' Feature dimension expected: ', config['circuit_width'],'.')

modelSavedPath = config['modelSavedPath']
resultOutputPath = config['resultOutputPath']

QKE_model = QKE_SVC(config['classical'], 
                    config['class_weight'], 
                    modelSavedPath = modelSavedPath,
                    gamma = config['gamma'],
                    C_class = config['C_class'],
                    alpha = config['alpha'],
                    C_quant = config['C_quant'],
                    single_mapping = config['single_mapping'],
                    pair_mapping = config['pair_mapping'],
                    interaction = config['interaction'],
                    circuit_width = config['circuit_width']
                   )

if config['load_model'] == False:
    #train model
    QKE_model.set_model(load = False, 
                        train_data = train_data,
                        train_labels = train_labels, 
                        fileName = fileName
                       )
else:
    #load_model
    modelName = modelSavedPath+'/model_'+fileName+'.sav'
    model = joblib.load(modelName)
    print('Model ' + modelName + ' loaded!')
    QKE_model.set_model(load = True, model = model)

    
#test
model_predictions, model_scores = QKE_model.test(test_data)

#update dataframe with prediction column
test_data.insert(np.shape(test_data)[1], 'predictedLabels', model_predictions)
test_data.insert(np.shape(test_data)[1], 'trueLabels', test_labels)
test_data.insert(np.shape(test_data)[1], 'scores', model_scores)

#save resulting dataframe
if not Path(resultOutputPath).exists():
    Path(resultOutputPath).mkdir(parents=True)
resultDataName = resultOutputPath + '/result_'+fileName+'.pkl'
test_data.to_pickle(resultDataName)
print('Result from applying the SVC model to the test set stored as:', resultDataName)

elapsed_time = time.time() - start_time
print("elapsed time=", elapsed_time)
