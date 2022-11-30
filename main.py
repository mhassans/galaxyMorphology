import joblib
import numpy as np
import pandas as pd
from QKE_SVC import QKE_SVC
from funcs import load_config, parse_args, prepare_dataframe, get_train_test, normalize_data, makeOutputFileName
from pathlib import Path
from sklearn.preprocessing import LabelBinarizer
import code

config = load_config(parse_args().config)
fileName = makeOutputFileName(config['classical'],
                              config['trainPlusTestSize']*(1-config['testSetSize'])
                             )

df = prepare_dataframe(trainPlusTestSize=config['trainPlusTestSize'])
train_data, train_labels_mtx, test_data, test_labels_mtx = get_train_test(df, testSetSize=config['testSetSize'])
train_data = normalize_data(train_data)
test_data = normalize_data(test_data)
classes = [0, 1]
lb = LabelBinarizer()
lb.fit(classes)
train_labels = lb.inverse_transform(train_labels_mtx)
test_labels = lb.inverse_transform(test_labels_mtx)

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
if config['classical']:
    model_predictions, model_scores = QKE_model.test(test_data)
else:
    model_predictions, model_scores = QKE_model.test(test_data, train_data)

#update dataframe with prediction column
test_data.insert(np.shape(test_data)[1], 'predictedLabels', model_predictions)
test_data.insert(np.shape(test_data)[1], 'trueLabels', test_labels)
test_data.insert(np.shape(test_data)[1], 'scores', model_scores)
test_data = pd.concat([test_data, test_labels_mtx], axis=1)
code.interact(local=locals())

#save resulting dataframe
if not Path(resultOutputPath).exists():
    Path(resultOutputPath).mkdir(parents=True)
resultDataName = resultOutputPath + '/result_'+fileName+'.pkl'
test_data.to_pickle(resultDataName)
print('Result from applying the SVC model to the test set stored as:', resultDataName)

#results = test_data_in_region.to_numpy()
#tracklet_type = config['tracklet_dataset']
#if config['classical']:
#    tracklet_type = 'classical_'+tracklet_type
#results_file = tracklet_type+'_predictions_'+str(config['num_train'])+'_'+str(config['num_test'])+'_events_reg_'+str(config['region_id'])+'_in_'+str(config['division'])
#
#np.save(results_file, results)
