import joblib
import numpy as np
import pandas as pd
from QKE_SVC import QKE_SVC
import code
from funcs import load_config, parse_args, prepare_dataframe, get_train_test, normalize_data

config, config_filename = load_config(parse_args().config)
df = prepare_dataframe()
train_data, train_labels, test_data, test_labels = get_train_test(df)
train_data = normalize_data(train_data)
test_data = normalize_data(test_data)

code.interact(local=locals())

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

    
