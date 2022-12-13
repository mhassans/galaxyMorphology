import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse


def load_config(file_path = None):
    path = Path(file_path)
    with open(path, 'r') as file:
        cfg = yaml.load(file, Loader = yaml.FullLoader)
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description = 'Load config file for SVM classification.')
    parser.add_argument('config', type = str, help = 'Path to config file.')

    return parser.parse_args()

def fix_ttype_feature(df, Ndigit=5):
    """
    TType is a mix of float and string. It also has a few values with '-' between digits, e.g. '5.292892456050-05'.
    This example could mean 5.29^+5, which is far from the range of other values! So such values are discarded here;
    i.e. TTypes with '-' in the last Ndigit are discarded.
    """
    df = df.astype({'TType': 'str'})
    df = df[~(df['TType'].str.find('-',-1*Ndigit)!=-1)] # remove TTypes with '-' in the last Ndigit
    df = df.astype({'TType': 'float'})
    return df

def prepare_dataframe(trainPlusTestSize):
    GZ1 = pd.read_csv('data/GalaxyZoo1/GZ1.csv') #Galaxy zoo data from data.galaxyzoo.org (unwanted columns removed)
    features = pd.read_csv('data/features/features.csv') 
                    #from sciencedirect.com/science/article/pii/S2213133719300757 (unwanted columns removed)
    features = features.rename(columns={'dr7objid':'OBJID'})
    features = features.drop_duplicates()
    df = pd.merge(features, GZ1, on='OBJID', how='inner')
    df = df[df.UNCERTAIN==0].drop(columns=['UNCERTAIN']) #remove Uncertain category, i.e. not elliptical nor spiral
    df = df[df.Error==0] #Keep successful CyMorph processes only. See kaggle.com/datasets/saurabhshahane/galaxy-classification
    df = df[(df['G2']>-6000) & (df['S']>-6000) & (df['A']>-6000) & (df['C']>-6000)] #discard outliers
    df = df.drop(['Error'], axis=1) 
    df = fix_ttype_feature(df)
    df = df.sample(n=trainPlusTestSize, random_state=1)
    df = df.reset_index(drop=True)
    return df

def get_train_test(df, n_splits=5, fold_idx=0):
    if fold_idx>=n_splits:
        print("ERROR: Fold index must be between 0 and n_splits-1.")
        sys.exit(1)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    indices = list(kf.split(df))
    train_id, test_id = indices[fold_idx]
    train = df.iloc[train_id]
    test = df.iloc[test_id]
    train_data = train.drop(['OBJID','SPIRAL','ELLIPTICAL'], axis=1)
    train_labels = train['SPIRAL']
    test_data = test.drop(['OBJID','SPIRAL','ELLIPTICAL'], axis=1)
    test_labels = test['SPIRAL']
    return train_data, train_labels, test_data, test_labels

def normalize_data(df):
    return ((df-df.min())/(df.max()-df.min()))

def makeOutputFileName(classical, trainSize):
    fileName = 'Class' if classical else 'Quant'
    fileName = fileName +'_trainSize' + str(int(trainSize))
    return fileName

def produceConfig(config, fileName):
    filePath = 'config/autoConfigs_'+fileName+'.yaml'
    with open(filePath, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    return filePath

def getConfigName(config):
    fileName = ''
    if config['classical']:
        fileName = 'Class'
        fileName += '-C' + str(config['C_class']).replace('.','p')
        fileName += '-gamma' + str(config['gamma']).replace('.','p')
    else:
        fileName = 'Quant'
        fileName += '-alpha' + str(config['alpha']).replace('.','p')
        fileName += '-C' + str(config['C_quant']).replace('.','p')
        fileName += '-singleMap' + str(config['single_mapping'])
        fileName += '-pairMap' + str(config['pair_mapping'])
        fileName += '-interaction' + config['interaction']
    
    fileName += '-weight'
    if (config['class_weight'] == None):
        fileName += 'None'
    elif (config['class_weight'] == 'balanced'):
        fileName += 'Balanced'
    else:
        print('ERROR: You need to change this part of the code to handle the file name when weights are other than None or balanced.')
        sys.exit(1)
    fileName += '-trainSizeADDTRAINSIZE' #+ str(int(trainSize))
    return fileName
