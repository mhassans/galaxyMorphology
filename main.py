import time
start_time = time.time()
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import pandas as pd
from QKE_SVC import QKE_SVC
from funcs import prepare_dataframe, get_train_test, normalize_data, setConfigName
from pathlib import Path
import sys
import yaml

try:
    config_file = sys.argv[1]
except IndexError:
    print("Please give a valid config file")
    exit()
try:
    with open(config_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
except EnvironmentError:
    print("Please give a valid config file")
    exit()
fileName = setConfigName(config)

availableFeatures = ['C', 'H', 'G2', 'S', 'A']#all available features in the datasets
excludedFeatures = config['excludedFeatures']#List of Features excluded from training. If empty, all features used.
usedFeatures = [ftr for ftr in availableFeatures if ftr not in excludedFeatures]
signalLabel = 'SPIRAL'

df = prepare_dataframe(trainPlusTestSize=config['trainPlusTestSize'], minOfK=config['minOfK'], signalLabel=signalLabel,\
                       excludedFeatures=excludedFeatures, balancedSampling=config['balancedSampling'])
train_data, train_labels, test_data, test_labels, train_extraInfo, test_extraInfo \
            = get_train_test(df, signalLabel, usedFeatures, n_splits=config['n_splits'], fold_idx=config['fold_idx'])
train_data = normalize_data(train_data)
test_data = normalize_data(test_data)
print('train_head\n', train_data.head())
n_features = np.shape(train_data)[1]
if not (config['circuit_width'] == n_features):
    raise AssertionError('Number of feautures is: ', n_features, '.',
    ' Feature dimension expected: ', config['circuit_width'],'.')

modelSavedPath = config['modelSavedPath'] + '/' + config['subDir'] + '/'
resultOutputPath = config['resultOutputPath'] + '/' + config['subDir'] + '/'

QKE_model = QKE_SVC(config['classical'], 
                    config['class_weight'], 
                    modelSavedPath = modelSavedPath,
                    entangleType = config['entangleType'],
                    nShots = config['nShots'],
                    RunOnIBMdevice = config['RunOnIBMdevice'],
                    corrMethod = config['corrMethod'],
                    gamma = config['gamma'],
                    C_class = config['C_class'],
                    alpha = config['alpha'],
                    alphaCorr = config['alphaCorr'],
                    C_quant = config['C_quant'],
                    data_map_func = config['data_map_func'],
                    interaction = config['interaction'],
                    circuit_width = config['circuit_width']
                   )

timeStampBeforeTraining = time.time()
print("Time elapsed before training starts: ", timeStampBeforeTraining - start_time, " seconds.")

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

timeStampAfterTrainingTime = time.time()
print("Training took ", timeStampAfterTrainingTime - timeStampBeforeTraining, " seconds.")
    
#test
model_predictions, model_scores = QKE_model.test(test_data)

timeStampAfterTesting = time.time()
print("Applying model to test set took ", timeStampAfterTesting - timeStampAfterTrainingTime, " seconds.")

#update dataframe with prediction column
test_data.insert(np.shape(test_data)[1], 'predictedLabels', model_predictions)
test_data.insert(np.shape(test_data)[1], 'trueLabels', test_labels)
test_data.insert(np.shape(test_data)[1], 'scores', model_scores)
test_data = pd.concat([test_extraInfo, test_data], axis=1)

timeStampAfterAddCol = time.time()
print("Adding results to dataframe took ", timeStampAfterAddCol - timeStampAfterTesting, " seconds.")

#save resulting dataframe
if not Path(resultOutputPath).exists():
    Path(resultOutputPath).mkdir(parents=True)
resultDataName = resultOutputPath + '/result_'+fileName+'.pkl'
test_data.to_pickle(resultDataName)
print('Result from applying the SVC model to the test set stored as:', resultDataName)
timeStampResultSaved = time.time()
print("Saving results dataframe took ", timeStampResultSaved - timeStampAfterAddCol," seconds.")

elapsed_time = time.time() - start_time
print("Running the code took ", elapsed_time, " seconds.")
print('######################################################')
print('ROC AUC: ', roc_auc_score(test_data['trueLabels'], test_data['scores']))
print('F1 score: ', f1_score(test_data['trueLabels'], test_data['predictedLabels']))
print('Accuracy: ', accuracy_score(test_data['trueLabels'], test_data['predictedLabels']))
