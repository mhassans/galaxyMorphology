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
    df = df.sample(n=trainPlusTestSize, random_state=33)
    df = df.reset_index(drop=True)
    return df

def get_train_test(df, testSetSize):
    train, test = train_test_split(df, test_size=testSetSize, random_state=42)
    train_data = train.drop(['OBJID','SPIRAL','ELLIPTICAL'], axis=1)
    train_labels_mtx = train[['ELLIPTICAL','SPIRAL']]
    test_data = test.drop(['OBJID','SPIRAL','ELLIPTICAL'], axis=1)
    test_labels_mtx = test[['ELLIPTICAL','SPIRAL']]
    return train_data, train_labels_mtx, test_data, test_labels_mtx

def normalize_data(df):
    return ((df-df.min())/(df.max()-df.min()))

def makeOutputFileName(classical, trainSize):
    fileName = 'Class' if classical else 'Quant'
    fileName = fileName +'_trainSize' + str(int(trainSize))
    return fileName
