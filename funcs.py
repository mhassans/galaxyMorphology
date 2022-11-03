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
    #returns the model and the name of the config run
    return cfg, path.stem

def parse_args():
    parser = argparse.ArgumentParser(description = 'Load config file for SVM classification.')
    parser.add_argument('config', type = str, help = 'Path to config file.')

    return parser.parse_args()

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

def normalize_data(df):
    return ((df-df.min())/(df.max()-df.min()))

