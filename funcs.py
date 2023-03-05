import yaml
import ast
import pandas as pd
from sklearn.model_selection import KFold
from functools import reduce
import numpy as np

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

def prepare_dataframe(trainPlusTestSize, minOfK):
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
    df = df[df['K']>minOfK]
    df = df.sample(n=trainPlusTestSize, random_state=1)
    df = df.reset_index(drop=True)
    return df

def get_train_test(df, n_splits=5, fold_idx=0):
    if fold_idx>=n_splits:
        print("ERROR: Fold index must be between 0 and n_splits-1.")
        exit()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    indices = list(kf.split(df))
    train_id, test_id = indices[fold_idx]
    train = df.iloc[train_id]
    test = df.iloc[test_id]
    features = ['C','A','S','H','G2']
    labels = 'SPIRAL'
    extraInfo = ['OBJID', 'TType','K']
    train_data = train[features]
    train_labels = train[labels]
    train_extraInfo = train[extraInfo]
    test_data = test[features]
    test_labels = test[labels]
    test_extraInfo = test[extraInfo]
    return train_data, train_labels, test_data, test_labels, train_extraInfo, test_extraInfo

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
    print('The following config file was produced:')
    print(filePath)
    return filePath

def setConfigName(config):
    fileName = ''
    if config['classical']:
        fileName = 'Class'
        fileName += '-C' + str(config['C_class']).replace('.','p')
        fileName += '-gamma' + str(config['gamma']).replace('.','p')
    else:
        fileName = 'Quant'
        fileName += '-alpha' + str(config['alpha']).replace('.','p')
        fileName += '-C' + str(config['C_quant']).replace('.','p')
        fileName += '-dataMapFunc'
        if (config['data_map_func'] == None):
            fileName += 'None'
        else:
            fileName += config['data_map_func'].__name__
        fileName += '-interaction' + 'and'.join(config['interaction'])
    
    fileName += '-weight'
    if (config['class_weight'] == None): # Here, 'class' has nothing to do with classical vs quantum.
        fileName += 'None'
    elif (config['class_weight'] == 'balanced'):
        fileName += 'Balanced'
    else:
        print('ERROR in funcs.py -> setConfigName function: You need to change this part of the code to set the file name when weights are other than None or balanced.')
        exit()
    testSetFraction = 1./config['n_splits']
    trainSetFraction = 1 - testSetFraction
    fileName += '-trainSize' + str(int(config['trainPlusTestSize']*trainSetFraction))
    fileName += '-testSize' + str(int(config['trainPlusTestSize']*testSetFraction))
    fileName += '-foldIdx' + str(config['fold_idx'])
    fileName += '-minOfK' + str(config['minOfK'])
    return fileName

def addMeanAndStdDev(df, classical):
    """
    Adds mean and std dev of metrics (roc, f1 score, and accuracy) and code run time as new df columns.
    Adds index information (weight, kernel, etc.) as new df columns.
    inputs:
        df: dataframe
        classical (boolean): whether it is for classical or quantum kernel
    """
    times = ["TimeToRun-fold{}".format(i) for i in range(5)]
    rocAUCs = ["rocAUC-fold{}".format(i) for i in range(5)]
    fOnes = ["F1score-fold{}".format(i) for i in range(5)]
    accuracies = ["Accuracy-fold{}".format(i) for i in range(5)]

    df["TimeToRun-mean"] = df[times].mean(axis=1)
    df["TimeToRun-std"] = df[times].std(axis=1)
    df["rocAUC-mean"] = df[rocAUCs].mean(axis=1)
    df["rocAUC-std"] = df[rocAUCs].std(axis=1)
    df["fOne-mean"] = df[fOnes].mean(axis=1)
    df["fOne-std"] = df[fOnes].std(axis=1)
    df["accuracy-mean"] = df[accuracies].mean(axis=1)
    df["accuracy-std"] = df[accuracies].std(axis=1)
    df['rocAUC-relError'] = df["rocAUC-std"]/df["rocAUC-mean"]
    df['fOne-relError'] = df["fOne-std"]/df["fOne-mean"]
    df['accuracy-relError'] = df["accuracy-std"]/df["accuracy-mean"]
    if classical:
        #Add index to column. Example of index: "('RBF', '1000000.0', 'scale', 'None')"
        df = df.assign(kernel = df.index.map(lambda x: str(ast.literal_eval(x)[0]))) #"ast" changes string back to tuple
        df = df.assign(C_class = df.index.map(lambda x: float(ast.literal_eval(x)[1])))
        df = df.assign(gamma = df.index.map(lambda x: str(ast.literal_eval(x)[2])))
        df = df.assign(weight = df.index.map(lambda x: str(ast.literal_eval(x)[3])))
    else:
        df = df.assign(alpha = df.index.map(lambda x: str(ast.literal_eval(x)[0]))) #"ast" changes string back to tuple
        df = df.assign(C_quant = df.index.map(lambda x: float(ast.literal_eval(x)[1])))
        df = df.assign(dataMapFunc = df.index.map(lambda x: str(ast.literal_eval(x)[2])))
        df = df.assign(interaction = df.index.map(lambda x: str(ast.literal_eval(x)[3])))
        df = df.assign(weight= df.index.map(lambda x: str(ast.literal_eval(x)[4])))
    return df

def dataMap_custom1(x):
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m*n, x)
    return coeff

def dataMap_custom2(x):
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m*n, 1 - x)
    return coeff

def dataMap_custom3(x):
    """
    x: array
    returns \prod_{i,j}(x[i]-x[j]) for all i & j in indices where i<j. If len(x) is 1 then returns x[0].
    """
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m * n, [(x[i] - x[j]) for i in range(len(x)) for j in range(i + 1, len(x))])
    return coeff

def dataMap_custom4(x):
    coeff = x[0] if len(x) == 1 else (1 - dataMap_custom3(x))
    return coeff
