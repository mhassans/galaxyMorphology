import joblib
import numpy as np
import pandas as pd
from QKE_SVC import QKE_SVC
from funcs import load_config, parse_args, prepare_dataframe, get_train_test, normalize_data

config, config_filename = load_config(parse_args().config)
df = prepare_dataframe(TrainPlusTestSize=1000)
train_data, train_labels, test_data, test_labels = get_train_test(df, testSize=0.5)
train_data = normalize_data(train_data)
test_data = normalize_data(test_data)

n_features = np.shape(train_data)[1]
if not (config['circuit_width'] == n_features):
    raise AssertionError('Number of feautures is: ', n_features, '.',
    ' Feature dimension expected: ', config['circuit_width'],'.')

QKE_model = QKE_SVC(config['classical'], 
                    config['class_weight'], 
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
                        from_config = config_filename
                       )
else:
    #load_model
    model = joblib.load(config['model_file'])
    QKE_model.set_model(load = True, model = model)

    
#test
if config['classical']:
    model_predictions, model_scores = QKE_model.test(test_data)
else:
    model_predictions, model_scores = QKE_model.test(test_data, train_data)

#update dataframe with prediction column
test_data.insert(np.shape(test_data)[1], 'predictedLabels', model_predictions)
test_data.insert(np.shape(test_data)[1], 'trueLables', test_labels)
test_data.insert(np.shape(test_data)[1], 'scores', model_scores)

#save resulting dataframe
test_data.to_pickle('output/test_data.pkl')

#results = test_data_in_region.to_numpy()
#tracklet_type = config['tracklet_dataset']
#if config['classical']:
#    tracklet_type = 'classical_'+tracklet_type
#results_file = tracklet_type+'_predictions_'+str(config['num_train'])+'_'+str(config['num_test'])+'_events_reg_'+str(config['region_id'])+'_in_'+str(config['division'])
#
#np.save(results_file, results)
