import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from QKE_SVC import QKE_SVC
from funcs import load_config, parse_args

def prepare_dataframe():
    GZ1 = pd.read_csv('data/GalaxyZoo1/GZ1.csv') #Galaxy zoo data from data.galaxyzoo.org (unwanted columns removed)
    features = pd.read_csv('data/features/features.csv') 
                    #from sciencedirect.com/science/article/pii/S2213133719300757 (unwanted columns removed)
    features = features.rename(columns={'dr7objid':'OBJID'})
    features = features.drop_duplicates()
    df = pd.merge(features, GZ1, on='OBJID', how='inner')
    df = df[df.UNCERTAIN==0].drop(columns=['UNCERTAIN']) #remove Uncertain category, i.e. not elliptical nor spiral
    df = df[df.Error==0] #Keep successful CyMorph processes only. See kaggle.com/datasets/saurabhshahane/galaxy-classification
    df = df[(df['G2']>-6000) & (df['S']>-6000) & (df['A']>-6000) & (df['C']>-6000)] #discard outliers
    df = df.drop(['Error', 'TType'], axis=1).reset_index(drop=True) #FIXME keep ttype? ttype is mix of string and float.
    return df

def get_train_test(df):
    train, test = train_test_split(df, test_size=0.2)
    train_data = train.drop(['OBJID','SPIRAL','ELLIPTICAL'], axis=1)
    train_labels = train['SPIRAL']
    test_data = test.drop(['OBJID','SPIRAL','ELLIPTICAL'], axis=1)
    test_labels = test['SPIRAL']
    return train_data, train_labels, test_data, test_labels

def main():
    #load configuration parameters defining the classification run
    config, config_filename = load_config(parse_args().config)
    df = prepare_dataframe()
    train_data, train_labels, test_data, test_labels = get_train_test(df)
    
    QKE_model = QKE_SVC(config['classical'], 
    config['class_weight'], 
    gamma = config['gamma'],
    C_class = config['C_class'],
    alpha = config['alpha'],
    C_quant = config['C_quant'],
    single_mapping = config['single_mapping'],
    pair_mapping = config['pair_mapping'],
    interaction = config['interaction'],
    circuit_width = config['circuit_width'])

if __name__=="__main__":
    main()
